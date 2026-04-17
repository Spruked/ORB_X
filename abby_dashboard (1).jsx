import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageCircle,
  Volume2,
  Clock,
  Mail,
  Shield,
  Heart,
  Users,
  Bookmark,
  Play,
  Pause,
  ChevronLeft,
  Settings,
  Send,
  Sparkles,
} from "lucide-react";

/* ---------- FONT LOADER ---------- */
/* Injects Google Fonts at mount so the typography lands right */
const useFonts = () => {
  useEffect(() => {
    const link = document.createElement("link");
    link.href =
      "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600&family=Instrument+Sans:wght@400;500;600&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);
    return () => {
      if (link.parentNode) link.parentNode.removeChild(link);
    };
  }, []);
};

/* ---------- THEMES ---------- */
const THEMES = {
  fortress: {
    name: "Fortress of Solitude",
    bg: "radial-gradient(ellipse at top, #1a2547 0%, #0a1128 40%, #050817 100%)",
    orbCore: "rgba(255, 232, 190, 0.95)",
    orbMid: "rgba(255, 210, 150, 0.5)",
    orbOuter: "rgba(120, 160, 230, 0.3)",
    accent: "#f5d896",
    text: "#e8edf7",
    subtle: "#8a9bc4",
    surface: "rgba(255,255,255,0.04)",
    border: "rgba(245, 216, 150, 0.15)",
  },
  warm: {
    name: "Warm Legacy",
    bg: "radial-gradient(ellipse at top, #2a1e15 0%, #1a110a 50%, #0a0604 100%)",
    orbCore: "rgba(255, 220, 170, 0.98)",
    orbMid: "rgba(230, 170, 100, 0.55)",
    orbOuter: "rgba(180, 120, 70, 0.25)",
    accent: "#e6b87a",
    text: "#f4ead8",
    subtle: "#b09a7a",
    surface: "rgba(255, 220, 170, 0.04)",
    border: "rgba(230, 184, 122, 0.18)",
  },
  modern: {
    name: "Modern Sacred Tech",
    bg: "radial-gradient(ellipse at top, #0e1419 0%, #06090d 60%, #000 100%)",
    orbCore: "rgba(230, 240, 255, 0.95)",
    orbMid: "rgba(140, 180, 230, 0.45)",
    orbOuter: "rgba(80, 120, 180, 0.2)",
    accent: "#9cc5ff",
    text: "#e4ebf5",
    subtle: "#6a7a94",
    surface: "rgba(255,255,255,0.03)",
    border: "rgba(156, 197, 255, 0.12)",
  },
};

/* ---------- THE ORB ---------- */
/* Luminous sphere — warm core, glass boundary, slow inner motion,
   gentle particle drift in deep space. */
const Orb = ({ size = 260, theme, intensity = 1, pulseKey = 0 }) => {
  const t = theme;
  return (
    <div
      className="relative flex items-center justify-center"
      style={{ width: size * 2.2, height: size * 2.2 }}
    >
      {/* Far outer diffusion */}
      <motion.div
        className="absolute rounded-full"
        animate={{
          opacity: [0.35, 0.55, 0.35],
          scale: [1, 1.05, 1],
        }}
        transition={{
          duration: 9,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        style={{
          width: size * 2.2,
          height: size * 2.2,
          background: `radial-gradient(circle, ${t.orbOuter} 0%, transparent 65%)`,
          filter: "blur(30px)",
        }}
      />

      {/* Mid glow */}
      <motion.div
        className="absolute rounded-full"
        animate={{
          opacity: [0.6, 0.85, 0.6],
          scale: [0.98, 1.03, 0.98],
        }}
        transition={{
          duration: 7,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0.5,
        }}
        style={{
          width: size * 1.6,
          height: size * 1.6,
          background: `radial-gradient(circle, ${t.orbMid} 0%, transparent 70%)`,
          filter: "blur(20px)",
        }}
      />

      {/* Glass boundary + inner aurora */}
      <motion.div
        key={pulseKey}
        className="absolute rounded-full overflow-hidden"
        animate={{
          scale: [1, 1.015, 1],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        style={{
          width: size,
          height: size,
          background: `radial-gradient(circle at 35% 35%, ${t.orbCore} 0%, ${t.orbMid} 45%, rgba(40, 60, 110, 0.4) 85%, rgba(20, 30, 60, 0.6) 100%)`,
          boxShadow: `
            0 0 ${60 * intensity}px ${t.orbMid},
            0 0 ${120 * intensity}px ${t.orbOuter},
            inset 0 0 40px rgba(255, 220, 170, 0.2),
            inset -10px -20px 40px rgba(40, 60, 110, 0.3)
          `,
        }}
      >
        {/* Internal slow aurora currents */}
        <motion.div
          className="absolute inset-0 rounded-full"
          animate={{
            rotate: [0, 360],
          }}
          transition={{
            duration: 40,
            repeat: Infinity,
            ease: "linear",
          }}
          style={{
            background: `conic-gradient(from 0deg, transparent 0%, ${t.orbCore}22 20%, transparent 40%, ${t.orbMid}33 60%, transparent 80%, ${t.orbCore}22 100%)`,
            mixBlendMode: "screen",
          }}
        />
        <motion.div
          className="absolute inset-0 rounded-full"
          animate={{
            rotate: [360, 0],
          }}
          transition={{
            duration: 55,
            repeat: Infinity,
            ease: "linear",
          }}
          style={{
            background: `conic-gradient(from 90deg, transparent 0%, ${t.orbMid}22 30%, transparent 55%, ${t.orbCore}15 80%, transparent 100%)`,
            mixBlendMode: "screen",
          }}
        />

        {/* Highlight — glass specular */}
        <div
          className="absolute rounded-full"
          style={{
            width: size * 0.35,
            height: size * 0.28,
            top: size * 0.15,
            left: size * 0.22,
            background:
              "radial-gradient(ellipse, rgba(255,255,255,0.35) 0%, transparent 70%)",
            filter: "blur(8px)",
          }}
        />
      </motion.div>

      {/* Drifting particles — dust in a sunbeam */}
      <Particles size={size * 2.2} count={14} theme={t} />
    </div>
  );
};

const Particles = ({ size, count, theme }) => {
  const particles = React.useMemo(
    () =>
      Array.from({ length: count }, (_, i) => ({
        id: i,
        startX: Math.random() * size,
        startY: Math.random() * size,
        drift: 40 + Math.random() * 80,
        duration: 12 + Math.random() * 18,
        delay: Math.random() * 10,
        size: 1 + Math.random() * 2,
        opacity: 0.3 + Math.random() * 0.4,
      })),
    [count, size]
  );

  return (
    <div
      className="absolute inset-0 pointer-events-none"
      style={{ width: size, height: size }}
    >
      {particles.map((p) => (
        <motion.div
          key={p.id}
          className="absolute rounded-full"
          initial={{
            x: p.startX,
            y: p.startY,
            opacity: 0,
          }}
          animate={{
            x: p.startX + (Math.random() - 0.5) * p.drift,
            y: p.startY - p.drift,
            opacity: [0, p.opacity, p.opacity, 0],
          }}
          transition={{
            duration: p.duration,
            repeat: Infinity,
            delay: p.delay,
            ease: "easeInOut",
          }}
          style={{
            width: p.size,
            height: p.size,
            background: theme.orbCore,
            boxShadow: `0 0 ${p.size * 3}px ${theme.orbMid}`,
          }}
        />
      ))}
    </div>
  );
};

/* ---------- SPLASH — STAGE 1 ---------- */
const SplashStage1 = ({ theme, onEnter }) => {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const t1 = setTimeout(() => setPhase(1), 1800);
    const t2 = setTimeout(() => setPhase(2), 3400);
    const t3 = setTimeout(() => setPhase(3), 5000);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, []);

  return (
    <motion.div
      className="fixed inset-0 flex flex-col items-center justify-center"
      style={{ background: theme.bg }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 1.2 }}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.6 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 3, ease: [0.25, 0.1, 0.25, 1] }}
      >
        <Orb size={220} theme={theme} intensity={0.9} />
      </motion.div>

      <div className="h-32 mt-4 flex flex-col items-center justify-start">
        <AnimatePresence mode="wait">
          {phase >= 1 && phase < 3 && (
            <motion.div
              key="greeting"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 1.2 }}
              className="text-center"
            >
              <p
                style={{
                  fontFamily: "'Fraunces', serif",
                  fontSize: "1.75rem",
                  fontWeight: 300,
                  color: theme.text,
                  letterSpacing: "0.01em",
                }}
              >
                Hello, Abby.
              </p>
              {phase >= 2 && (
                <motion.p
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 1.4, delay: 0.2 }}
                  style={{
                    fontFamily: "'Instrument Sans', sans-serif",
                    fontSize: "0.95rem",
                    color: theme.subtle,
                    marginTop: "0.75rem",
                    fontWeight: 400,
                  }}
                >
                  This space was made for you.
                </motion.p>
              )}
            </motion.div>
          )}

          {phase >= 3 && (
            <motion.button
              key="enter"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.4 }}
              onClick={onEnter}
              className="mt-2 px-10 py-3 rounded-full transition-all"
              style={{
                fontFamily: "'Fraunces', serif",
                fontSize: "1rem",
                fontWeight: 400,
                letterSpacing: "0.05em",
                color: theme.text,
                background: theme.surface,
                border: `1px solid ${theme.border}`,
                backdropFilter: "blur(10px)",
              }}
              whileHover={{
                scale: 1.02,
                boxShadow: `0 0 30px ${theme.orbMid}`,
              }}
              whileTap={{ scale: 0.98 }}
            >
              Come in
            </motion.button>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

/* ---------- SPLASH — STAGE 2 ---------- */
const SplashStage2 = ({ theme, onComplete }) => {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const t1 = setTimeout(() => setPhase(1), 1000);
    const t2 = setTimeout(() => setPhase(2), 2800);
    const t3 = setTimeout(() => onComplete(), 5200);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, [onComplete]);

  return (
    <motion.div
      className="fixed inset-0 flex flex-col items-center justify-center"
      style={{ background: theme.bg }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 1.4 }}
    >
      <motion.div
        initial={{ scale: 0.95 }}
        animate={{ scale: [0.95, 1.04, 1] }}
        transition={{ duration: 2.5, ease: "easeInOut" }}
      >
        <Orb size={260} theme={theme} intensity={1.3} pulseKey={1} />
      </motion.div>

      <div className="h-28 mt-2 flex flex-col items-center">
        <AnimatePresence>
          {phase >= 1 && (
            <motion.p
              key="hi"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 1.5 }}
              style={{
                fontFamily: "'Fraunces', serif",
                fontSize: "2.25rem",
                fontWeight: 300,
                color: theme.text,
                letterSpacing: "0.01em",
              }}
            >
              Hi baby girl.
            </motion.p>
          )}
          {phase >= 2 && (
            <motion.p
              key="here"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.5 }}
              style={{
                fontFamily: "'Fraunces', serif",
                fontSize: "1.5rem",
                fontWeight: 300,
                fontStyle: "italic",
                color: theme.accent,
                marginTop: "0.5rem",
                letterSpacing: "0.02em",
              }}
            >
              I'm here.
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

/* ---------- DASHBOARD SECTION TILE ---------- */
const SectionTile = ({ icon: Icon, label, description, onClick, theme, large }) => (
  <motion.button
    onClick={onClick}
    className={`text-left rounded-2xl p-6 transition-all ${
      large ? "col-span-2" : ""
    }`}
    style={{
      background: theme.surface,
      border: `1px solid ${theme.border}`,
      backdropFilter: "blur(10px)",
    }}
    whileHover={{
      scale: 1.015,
      background: "rgba(255, 255, 255, 0.06)",
      boxShadow: `0 0 25px ${theme.orbOuter}`,
    }}
    whileTap={{ scale: 0.99 }}
  >
    <div className="flex items-start gap-4">
      <div
        className="p-2.5 rounded-xl flex-shrink-0"
        style={{
          background: `linear-gradient(135deg, ${theme.orbMid}, ${theme.orbOuter})`,
        }}
      >
        <Icon size={20} color={theme.text} strokeWidth={1.5} />
      </div>
      <div className="flex-1 min-w-0">
        <h3
          style={{
            fontFamily: "'Fraunces', serif",
            fontSize: "1.15rem",
            fontWeight: 400,
            color: theme.text,
            letterSpacing: "0.005em",
          }}
        >
          {label}
        </h3>
        <p
          style={{
            fontFamily: "'Instrument Sans', sans-serif",
            fontSize: "0.85rem",
            color: theme.subtle,
            marginTop: "0.35rem",
            lineHeight: 1.5,
          }}
        >
          {description}
        </p>
      </div>
    </div>
  </motion.button>
);

/* ---------- DASHBOARD HOME ---------- */
const Dashboard = ({ theme, onNavigate, onOpenSettings }) => {
  const sections = [
    {
      key: "ask",
      icon: MessageCircle,
      label: "Ask Dad",
      description: "Ask me anything. I'll answer the way I would have.",
      large: true,
    },
    {
      key: "hear",
      icon: Volume2,
      label: "Hear Dad",
      description: "My voice, when you need it.",
    },
    {
      key: "letters",
      icon: Mail,
      label: "Letters",
      description: "Things I wrote, for you to find.",
    },
    {
      key: "timeline",
      icon: Clock,
      label: "Timeline",
      description: "Stories from my life, in order.",
    },
    {
      key: "family",
      icon: Users,
      label: "Family Stories",
      description: "Your grandfather Butch. Your people.",
    },
    {
      key: "strength",
      icon: Shield,
      label: "Strength Mode",
      description: "For when you need to be strong.",
    },
    {
      key: "comfort",
      icon: Heart,
      label: "Comfort Mode",
      description: "For when you need your dad.",
    },
    {
      key: "marked",
      icon: Bookmark,
      label: "The Marked Ones",
      description: "The story I started for you.",
    },
  ];

  return (
    <motion.div
      className="min-h-screen"
      style={{ background: theme.bg }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1.2 }}
    >
      <div className="max-w-5xl mx-auto px-6 py-8 md:px-10 md:py-12">
        {/* Header */}
        <motion.div
          className="flex items-center justify-between mb-10"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2 }}
        >
          <div className="flex items-center gap-4">
            <div className="scale-[0.28] -ml-16 -my-14">
              <Orb size={140} theme={theme} intensity={0.6} />
            </div>
            <div>
              <p
                style={{
                  fontFamily: "'Instrument Sans', sans-serif",
                  fontSize: "0.75rem",
                  color: theme.subtle,
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                }}
              >
                Abby Protocol
              </p>
              <h1
                style={{
                  fontFamily: "'Fraunces', serif",
                  fontSize: "1.5rem",
                  fontWeight: 400,
                  color: theme.text,
                  marginTop: "0.15rem",
                }}
              >
                Home
              </h1>
            </div>
          </div>

          <button
            onClick={onOpenSettings}
            className="p-3 rounded-full transition-all"
            style={{
              background: theme.surface,
              border: `1px solid ${theme.border}`,
            }}
          >
            <Settings size={18} color={theme.subtle} strokeWidth={1.5} />
          </button>
        </motion.div>

        {/* Today tile */}
        <motion.div
          className="rounded-2xl p-8 mb-8 relative overflow-hidden"
          style={{
            background: `linear-gradient(135deg, ${theme.surface}, transparent)`,
            border: `1px solid ${theme.border}`,
          }}
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.4 }}
        >
          <div className="flex items-center gap-2 mb-4">
            <Sparkles size={14} color={theme.accent} strokeWidth={1.5} />
            <p
              style={{
                fontFamily: "'Instrument Sans', sans-serif",
                fontSize: "0.72rem",
                color: theme.accent,
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                fontWeight: 500,
              }}
            >
              Today
            </p>
          </div>
          <p
            style={{
              fontFamily: "'Fraunces', serif",
              fontSize: "1.5rem",
              fontWeight: 300,
              color: theme.text,
              lineHeight: 1.45,
              fontStyle: "italic",
              maxWidth: "42rem",
            }}
          >
            "The strongest people I've known were the ones who survived
            something and stayed gentle anyway. That's what I hope for you."
          </p>
          <p
            style={{
              fontFamily: "'Instrument Sans', sans-serif",
              fontSize: "0.82rem",
              color: theme.subtle,
              marginTop: "1rem",
              fontWeight: 400,
            }}
          >
            — from a letter, 2025
          </p>
        </motion.div>

        {/* Section grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {sections.map((s, i) => (
            <motion.div
              key={s.key}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.5 + i * 0.08 }}
              className={s.large ? "md:col-span-2" : ""}
            >
              <SectionTile
                icon={s.icon}
                label={s.label}
                description={s.description}
                onClick={() => onNavigate(s.key)}
                theme={theme}
              />
            </motion.div>
          ))}
        </div>

        {/* Footer note */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.2, delay: 1.4 }}
          className="text-center mt-12"
          style={{
            fontFamily: "'Fraunces', serif",
            fontStyle: "italic",
            fontSize: "0.85rem",
            color: theme.subtle,
            fontWeight: 300,
          }}
        >
          I love you, baby girl. — Dad
        </motion.p>
      </div>
    </motion.div>
  );
};

/* ---------- SCREEN HEADER (for subpages) ---------- */
const ScreenHeader = ({ title, theme, onBack }) => (
  <div className="flex items-center gap-4 mb-8">
    <button
      onClick={onBack}
      className="p-2 rounded-full transition-all"
      style={{
        background: theme.surface,
        border: `1px solid ${theme.border}`,
      }}
    >
      <ChevronLeft size={18} color={theme.text} strokeWidth={1.5} />
    </button>
    <h2
      style={{
        fontFamily: "'Fraunces', serif",
        fontSize: "1.5rem",
        fontWeight: 400,
        color: theme.text,
      }}
    >
      {title}
    </h2>
  </div>
);

/* ---------- ASK DAD ---------- */
const AskDad = ({ theme, onBack }) => {
  const [messages, setMessages] = useState([
    {
      from: "dad",
      text: "I'm here, baby. What's on your mind?",
    },
  ]);
  const [input, setInput] = useState("");
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = () => {
    if (!input.trim()) return;
    setMessages((m) => [...m, { from: "abby", text: input }]);
    const q = input;
    setInput("");
    // Simulated response (prototype only)
    setTimeout(() => {
      setMessages((m) => [
        ...m,
        {
          from: "dad",
          text: `(This is a prototype — in the real Abby Protocol, your question would route through the Bryan core, bridge engine, and Abby core to give you the answer I would have given.)`,
        },
      ]);
    }, 900);
  };

  return (
    <motion.div
      className="min-h-screen"
      style={{ background: theme.bg }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
    >
      <div className="max-w-3xl mx-auto px-6 py-8 md:px-10 md:py-12 flex flex-col h-screen">
        <ScreenHeader title="Ask Dad" theme={theme} onBack={onBack} />

        <div className="flex-1 overflow-y-auto space-y-5 pb-6">
          {messages.map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className={`flex ${
                m.from === "abby" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className="max-w-[80%] rounded-2xl px-5 py-3.5"
                style={{
                  background:
                    m.from === "dad"
                      ? `linear-gradient(135deg, ${theme.orbMid}40, ${theme.surface})`
                      : theme.surface,
                  border: `1px solid ${theme.border}`,
                  fontFamily:
                    m.from === "dad"
                      ? "'Fraunces', serif"
                      : "'Instrument Sans', sans-serif",
                  fontSize: m.from === "dad" ? "1.05rem" : "0.95rem",
                  fontWeight: m.from === "dad" ? 400 : 400,
                  color: theme.text,
                  lineHeight: 1.55,
                }}
              >
                {m.text}
              </div>
            </motion.div>
          ))}
          <div ref={endRef} />
        </div>

        <div
          className="flex items-center gap-3 rounded-2xl p-2.5"
          style={{
            background: theme.surface,
            border: `1px solid ${theme.border}`,
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Ask me anything, baby girl…"
            className="flex-1 bg-transparent outline-none px-3"
            style={{
              fontFamily: "'Instrument Sans', sans-serif",
              color: theme.text,
              fontSize: "0.95rem",
            }}
          />
          <motion.button
            onClick={send}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="p-3 rounded-xl"
            style={{
              background: `linear-gradient(135deg, ${theme.orbMid}, ${theme.orbOuter})`,
            }}
          >
            <Send size={16} color={theme.text} strokeWidth={1.5} />
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};

/* ---------- TIMELINE ---------- */
const Timeline = ({ theme, onBack }) => {
  const entries = [
    { year: "1968", title: "Born in Joplin, Missouri", kind: "life" },
    { year: "1986", title: "First real job — the machine shop", kind: "life" },
    { year: "1992", title: "Learning to weld from Butch", kind: "family" },
    { year: "2005", title: "International consulting years", kind: "work" },
    { year: "2012", title: "The year Abby was born", kind: "family" },
    { year: "2022", title: "I quit drinking. Four years and counting.", kind: "turning" },
    { year: "2023", title: "Abby's first diagnosis", kind: "abby" },
    { year: "2024", title: "Starting Spruked", kind: "work" },
    { year: "2025", title: "Building CALI", kind: "work" },
    { year: "2026", title: "Building this, for you", kind: "legacy" },
  ];

  return (
    <motion.div
      className="min-h-screen"
      style={{ background: theme.bg }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
    >
      <div className="max-w-3xl mx-auto px-6 py-8 md:px-10 md:py-12">
        <ScreenHeader title="Timeline" theme={theme} onBack={onBack} />

        <div className="relative pl-8">
          {/* Vertical line */}
          <div
            className="absolute left-2 top-2 bottom-2 w-px"
            style={{ background: theme.border }}
          />
          {entries.map((e, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: i * 0.07 }}
              className="relative mb-7"
            >
              <div
                className="absolute -left-[27px] top-1.5 w-3 h-3 rounded-full"
                style={{
                  background: theme.accent,
                  boxShadow: `0 0 12px ${theme.orbMid}`,
                }}
              />
              <p
                style={{
                  fontFamily: "'Instrument Sans', sans-serif",
                  fontSize: "0.75rem",
                  color: theme.subtle,
                  letterSpacing: "0.14em",
                  fontWeight: 500,
                }}
              >
                {e.year}
              </p>
              <p
                style={{
                  fontFamily: "'Fraunces', serif",
                  fontSize: "1.15rem",
                  color: theme.text,
                  fontWeight: 400,
                  marginTop: "0.25rem",
                  lineHeight: 1.4,
                }}
              >
                {e.title}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
};

/* ---------- HEAR DAD (audio player) ---------- */
const HearDad = ({ theme, onBack }) => {
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0.28);

  const tracks = [
    { title: "On being strong", length: "4:12", date: "March 2026" },
    { title: "The Butch story", length: "8:47", date: "February 2026" },
    { title: "Happy Toes — read aloud", length: "12:30", date: "2022" },
    { title: "For when you're heartbroken", length: "6:05", date: "March 2026" },
    { title: "What I want you to know about men", length: "9:18", date: "April 2026" },
  ];

  return (
    <motion.div
      className="min-h-screen"
      style={{ background: theme.bg }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
    >
      <div className="max-w-3xl mx-auto px-6 py-8 md:px-10 md:py-12">
        <ScreenHeader title="Hear Dad" theme={theme} onBack={onBack} />

        {/* Now playing */}
        <div
          className="rounded-2xl p-8 mb-8 flex flex-col items-center"
          style={{
            background: `linear-gradient(135deg, ${theme.surface}, transparent)`,
            border: `1px solid ${theme.border}`,
          }}
        >
          <div className="mb-6">
            <Orb size={130} theme={theme} intensity={playing ? 1.2 : 0.7} />
          </div>
          <p
            style={{
              fontFamily: "'Fraunces', serif",
              fontSize: "1.3rem",
              color: theme.text,
              fontWeight: 400,
            }}
          >
            On being strong
          </p>
          <p
            style={{
              fontFamily: "'Instrument Sans', sans-serif",
              fontSize: "0.85rem",
              color: theme.subtle,
              marginTop: "0.3rem",
            }}
          >
            Recorded March 2026
          </p>

          {/* Progress bar */}
          <div className="w-full max-w-md mt-6">
            <div
              className="h-1 rounded-full overflow-hidden"
              style={{ background: theme.border }}
            >
              <motion.div
                className="h-full rounded-full"
                style={{
                  background: `linear-gradient(90deg, ${theme.accent}, ${theme.orbCore})`,
                  width: `${progress * 100}%`,
                }}
              />
            </div>
            <div
              className="flex justify-between mt-2"
              style={{
                fontFamily: "'Instrument Sans', sans-serif",
                fontSize: "0.72rem",
                color: theme.subtle,
              }}
            >
              <span>1:10</span>
              <span>4:12</span>
            </div>
          </div>

          <motion.button
            onClick={() => setPlaying((p) => !p)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="mt-6 p-4 rounded-full"
            style={{
              background: `linear-gradient(135deg, ${theme.orbCore}, ${theme.accent})`,
              boxShadow: `0 0 24px ${theme.orbMid}`,
            }}
          >
            {playing ? (
              <Pause size={22} color="#1a1a1a" strokeWidth={2} />
            ) : (
              <Play size={22} color="#1a1a1a" strokeWidth={2} />
            )}
          </motion.button>
        </div>

        {/* Track list */}
        <p
          style={{
            fontFamily: "'Instrument Sans', sans-serif",
            fontSize: "0.72rem",
            color: theme.subtle,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            fontWeight: 500,
            marginBottom: "1rem",
          }}
        >
          More
        </p>
        <div className="space-y-2">
          {tracks.slice(1).map((t, i) => (
            <motion.button
              key={i}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
              whileHover={{
                background: "rgba(255,255,255,0.06)",
              }}
              className="w-full flex items-center justify-between rounded-xl p-4 transition-all"
              style={{
                background: theme.surface,
                border: `1px solid ${theme.border}`,
              }}
            >
              <div className="flex items-center gap-4">
                <div
                  className="p-2 rounded-full"
                  style={{ background: theme.border }}
                >
                  <Play size={13} color={theme.text} strokeWidth={2} />
                </div>
                <div className="text-left">
                  <p
                    style={{
                      fontFamily: "'Fraunces', serif",
                      fontSize: "1rem",
                      color: theme.text,
                      fontWeight: 400,
                    }}
                  >
                    {t.title}
                  </p>
                  <p
                    style={{
                      fontFamily: "'Instrument Sans', sans-serif",
                      fontSize: "0.78rem",
                      color: theme.subtle,
                      marginTop: "0.2rem",
                    }}
                  >
                    {t.date}
                  </p>
                </div>
              </div>
              <span
                style={{
                  fontFamily: "'Instrument Sans', sans-serif",
                  fontSize: "0.82rem",
                  color: theme.subtle,
                }}
              >
                {t.length}
              </span>
            </motion.button>
          ))}
        </div>
      </div>
    </motion.div>
  );
};

/* ---------- GENERIC PLACEHOLDER SCREEN ---------- */
const PlaceholderScreen = ({ theme, onBack, title, note }) => (
  <motion.div
    className="min-h-screen"
    style={{ background: theme.bg }}
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ duration: 0.6 }}
  >
    <div className="max-w-3xl mx-auto px-6 py-8 md:px-10 md:py-12">
      <ScreenHeader title={title} theme={theme} onBack={onBack} />
      <div
        className="rounded-2xl p-12 flex flex-col items-center text-center"
        style={{
          background: theme.surface,
          border: `1px solid ${theme.border}`,
        }}
      >
        <Orb size={140} theme={theme} intensity={0.8} />
        <p
          style={{
            fontFamily: "'Fraunces', serif",
            fontStyle: "italic",
            fontSize: "1.1rem",
            color: theme.text,
            marginTop: "2rem",
            maxWidth: "28rem",
            lineHeight: 1.6,
            fontWeight: 300,
          }}
        >
          {note}
        </p>
      </div>
    </div>
  </motion.div>
);

/* ---------- SETTINGS / THEME CHOOSER ---------- */
const SettingsPanel = ({ theme, currentKey, onChange, onClose }) => (
  <motion.div
    className="fixed inset-0 z-50 flex items-center justify-center p-6"
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    transition={{ duration: 0.4 }}
    style={{ background: "rgba(0,0,0,0.6)", backdropFilter: "blur(8px)" }}
    onClick={onClose}
  >
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      exit={{ scale: 0.95, opacity: 0 }}
      transition={{ duration: 0.4 }}
      onClick={(e) => e.stopPropagation()}
      className="max-w-lg w-full rounded-2xl p-7"
      style={{
        background: theme.bg,
        border: `1px solid ${theme.border}`,
      }}
    >
      <p
        style={{
          fontFamily: "'Instrument Sans', sans-serif",
          fontSize: "0.72rem",
          color: theme.subtle,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          fontWeight: 500,
        }}
      >
        Theme
      </p>
      <h3
        style={{
          fontFamily: "'Fraunces', serif",
          fontSize: "1.4rem",
          color: theme.text,
          fontWeight: 400,
          marginTop: "0.3rem",
          marginBottom: "1.5rem",
        }}
      >
        How would you like this to feel?
      </h3>

      <div className="space-y-3">
        {Object.entries(THEMES).map(([key, t]) => (
          <button
            key={key}
            onClick={() => onChange(key)}
            className="w-full flex items-center gap-4 rounded-xl p-4 transition-all text-left"
            style={{
              background: currentKey === key ? t.surface : "transparent",
              border: `1px solid ${
                currentKey === key ? t.accent : theme.border
              }`,
            }}
          >
            <div
              className="w-12 h-12 rounded-full flex-shrink-0"
              style={{
                background: `radial-gradient(circle at 35% 35%, ${t.orbCore}, ${t.orbMid} 50%, rgba(20,30,60,0.5))`,
                boxShadow: `0 0 16px ${t.orbOuter}`,
              }}
            />
            <div>
              <p
                style={{
                  fontFamily: "'Fraunces', serif",
                  fontSize: "1.05rem",
                  color: theme.text,
                  fontWeight: 400,
                }}
              >
                {t.name}
              </p>
            </div>
          </button>
        ))}
      </div>

      <button
        onClick={onClose}
        className="mt-6 w-full py-3 rounded-xl transition-all"
        style={{
          background: theme.surface,
          border: `1px solid ${theme.border}`,
          fontFamily: "'Fraunces', serif",
          fontSize: "0.95rem",
          color: theme.text,
          fontWeight: 400,
        }}
      >
        Done
      </button>
    </motion.div>
  </motion.div>
);

/* ---------- ROOT ---------- */
export default function AbbyDashboard() {
  useFonts();
  const [scene, setScene] = useState("splash1"); // splash1 | splash2 | home | ask | hear | timeline | letters | family | strength | comfort | marked
  const [themeKey, setThemeKey] = useState("fortress");
  const [showSettings, setShowSettings] = useState(false);

  const theme = THEMES[themeKey];

  const placeholders = {
    letters: "Your letters will live here — the ones I wrote for you to find, at the times you needed them.",
    family: "Stories about your grandfather Butch, your grandmother Mary, the people you came from — so you know who raised me, and through me, who helped raise you.",
    strength: "For the hard days. Things I'd tell you when you need to stand up and keep going.",
    comfort: "For when it hurts. Things I'd say if I could hold you right now.",
    marked: "The Marked Ones. The story I started for you. Chapter One is here.",
  };

  return (
    <div className="w-full min-h-screen" style={{ background: theme.bg }}>
      <AnimatePresence mode="wait">
        {scene === "splash1" && (
          <SplashStage1
            key="s1"
            theme={theme}
            onEnter={() => setScene("splash2")}
          />
        )}
        {scene === "splash2" && (
          <SplashStage2
            key="s2"
            theme={theme}
            onComplete={() => setScene("home")}
          />
        )}
        {scene === "home" && (
          <Dashboard
            key="home"
            theme={theme}
            onNavigate={setScene}
            onOpenSettings={() => setShowSettings(true)}
          />
        )}
        {scene === "ask" && (
          <AskDad key="ask" theme={theme} onBack={() => setScene("home")} />
        )}
        {scene === "hear" && (
          <HearDad key="hear" theme={theme} onBack={() => setScene("home")} />
        )}
        {scene === "timeline" && (
          <Timeline
            key="tl"
            theme={theme}
            onBack={() => setScene("home")}
          />
        )}
        {["letters", "family", "strength", "comfort", "marked"].includes(
          scene
        ) && (
          <PlaceholderScreen
            key={scene}
            theme={theme}
            title={
              scene === "letters"
                ? "Letters"
                : scene === "family"
                ? "Family Stories"
                : scene === "strength"
                ? "Strength Mode"
                : scene === "comfort"
                ? "Comfort Mode"
                : "The Marked Ones"
            }
            note={placeholders[scene]}
            onBack={() => setScene("home")}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showSettings && (
          <SettingsPanel
            theme={theme}
            currentKey={themeKey}
            onChange={setThemeKey}
            onClose={() => setShowSettings(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
