import { useEffect, useRef, useState } from "react";

const FRAME_COUNT = 64;
const RESPONSE = 0.26; // per-60fps-frame catch-up factor (~35ms to settle)
const DEADZONE_IN = 0.12; // of min(viewport w, h): look into the viewer's eyes
const DEADZONE_OUT = 0.14; // small hysteresis so it never flickers at the edge
const TAU = Math.PI * 2;

// Shortest-path circular lerp
function lerpAngle(a, b, t) {
    let d = (b - a) % TAU;
    if (d > Math.PI) d -= TAU;
    if (d < -Math.PI) d += TAU;
    return a + d * t;
}

// f00 = UP, indices grow clockwise on screen (same direction atan2 grows when y points down)
function angleToIndex(angle) {
    const turns = (angle + Math.PI / 2) / TAU; // 0 at UP
    return ((Math.round(turns * FRAME_COUNT) % FRAME_COUNT) + FRAME_COUNT) % FRAME_COUNT;
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.decoding = "async";
        img.onload = () => (img.decode ? img.decode().catch(() => { }).then(() => resolve(img)) : resolve(img));
        img.onerror = reject;
        img.src = src;
    });
}

export default function CharacterHero() {
    const canvasRef = useRef(null);
    const [meta, setMeta] = useState(null);
    const [ready, setReady] = useState(false);

    // 1. meta (size, exact background colour, face centre)
    useEffect(() => {
        fetch("/frames/meta.json")
            .then((r) => r.json())
            .then(setMeta)
            .catch(() => setMeta({ width: 1080, height: 1920, background: "#000000", faceCenter: { x: 0.5, y: 0.38 } }));
    }, []);

    // 2. preload + render loop
    useEffect(() => {
        if (!meta) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d", { alpha: false });
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        canvas.width = meta.width;
        canvas.height = meta.height;

        const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        let cancelled = false;
        let raf = 0;
        let frames = [];
        let center = null;

        const pointer = { x: window.innerWidth / 2, y: window.innerHeight / 2, active: false };
        const onMove = (e) => {
            pointer.x = e.clientX;
            pointer.y = e.clientY;
            pointer.active = true;
        };
        const onLeave = () => (pointer.active = false);
        window.addEventListener("pointermove", onMove, { passive: true });
        document.documentElement.addEventListener("pointerleave", onLeave);
        window.addEventListener("blur", onLeave);

        let angle = -Math.PI / 2;
        let inDead = true;
        let drawn = null; // what's currently on the canvas: frame index, or "center"
        let last = performance.now();

        const draw = (key) => {
            if (key === drawn) return;
            const img = key === "center" ? center : frames[key];
            if (!img) return;
            // One opaque frame, replacing everything. No blending, no ghosting.
            ctx.globalCompositeOperation = "copy";
            ctx.globalAlpha = 1;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            drawn = key;
        };

        const tick = (now) => {
            raf = requestAnimationFrame(tick);
            const dt = Math.min(now - last, 50);
            last = now;

            const rect = canvas.getBoundingClientRect();
            const fx = rect.left + rect.width * meta.faceCenter.x;
            const fy = rect.top + rect.height * meta.faceCenter.y;
            const dx = pointer.x - fx;
            const dy = pointer.y - fy;
            const dist = Math.hypot(dx, dy);
            const radius = Math.min(window.innerWidth, window.innerHeight);

            if (reduceMotion || !pointer.active) inDead = true;
            else if (inDead && dist > radius * DEADZONE_OUT) inDead = false;
            else if (!inDead && dist < radius * DEADZONE_IN) inDead = true;

            if (inDead) {
                draw("center");
                return;
            }
            // frame-rate independent version of lerp(a, b, 0.26)
            const k = 1 - Math.pow(1 - RESPONSE, dt / (1000 / 60));
            angle = lerpAngle(angle, Math.atan2(dy, dx), k);
            draw(angleToIndex(angle));
        };

        (async () => {
            const base = "/frames/";
            const names = Array.from({ length: FRAME_COUNT }, (_, i) => `${base}f${String(i).padStart(2, "0")}.webp`);
            // center first so the hero shows something immediately
            center = await loadImage(`${base}center.webp`);
            if (cancelled) return;
            draw("center");
            setReady(true);
            frames = await Promise.all(names.map(loadImage));
            if (cancelled) return;
            last = performance.now();
            raf = requestAnimationFrame(tick);
        })().catch((err) => console.error("Frame preload failed", err));

        return () => {
            cancelled = true;
            cancelAnimationFrame(raf);
            window.removeEventListener("pointermove", onMove);
            document.documentElement.removeEventListener("pointerleave", onLeave);
            window.removeEventListener("blur", onLeave);
        };
    }, [meta]);

    const bg = meta?.background ?? "#000";

    return (
        <section
            className="relative isolate flex h-screen min-h-160 w-full items-end justify-center overflow-hidden"
            style={{ backgroundColor: bg }}
        >
            {/* No perspective / rotateX / rotateY anywhere: the canvas never moves, only pixels inside it change. */}
            <canvas
                ref={canvasRef}
                aria-label="Illustrated portrait that follows your cursor"
                role="img"
                className={`h-full max-h-screen w-auto max-w-full object-contain transition-opacity duration-700 ${ready ? "opacity-100" : "opacity-0"
                    }`}
                style={{ backgroundColor: bg }}
            />

            <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between px-6 pb-8 sm:px-12 sm:pb-12">
                <h1
                    className="text-[clamp(3rem,10vw,9rem)] leading-[0.9] tracking-tight text-[#386641]/95"
                    style={{ fontFamily: "'Cormorant Garamond', 'Times New Roman', serif", fontWeight: 500 }}
                >
                    Riya <br />
                    <span>Thakor</span>
                </h1>
                <p
                    className="max-w-[16rem] pb-2 text-right text-sm leading-relaxed text-[#386641] sm:text-md font-semibold"
                    style={{ fontFamily: "'Manrope', system-ui, sans-serif" }}
                >
                    Frontend developer building considered, fast interfaces in React.
                </p>
            </div>
        </section>
    );
}