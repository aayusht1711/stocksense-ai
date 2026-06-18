"use client";

import { useEffect, useState, useRef } from "react";
import dynamic from "next/dynamic";
import { motion, Variants, AnimatePresence, useMotionValue, useTransform } from "framer-motion";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Activity, TrendingUp, DollarSign, Crosshair, ArrowRight, ArrowUpRight, ArrowDownRight, Bitcoin, Briefcase, Users, MessageSquare, History, BellRing, GripHorizontal } from "lucide-react";
import Link from "next/link";
import VoiceCopilot from "../components/VoiceCopilot";
import confetti from "canvas-confetti";
import useSound from "use-sound";

// Dnd Kit Imports
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors } from "@dnd-kit/core";
import { arrayMove, SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy, useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

// Dynamically import the 3D scene to prevent SSR issues with WebGL
const Scene = dynamic(() => import("../components/Scene"), { ssr: false });

// Sound effect files (we will use generic URLs or silent fallbacks for now, since we don't have local mp3s)
const BEEP_SOUND = "https://actions.google.com/sounds/v1/ui/beep_short.ogg";
const CASH_SOUND = "https://actions.google.com/sounds/v1/science_fiction/computer_error_or_power_down.ogg"; // Fallback sci-fi sound

// Framer Motion Animation Variants
const containerVariants: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.15, delayChildren: 0.2 } }
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 30 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 70, damping: 15 } }
};

// Holographic Parallax Card Wrapper (Sortable)
function SortableHolographicCard({ id, children, className }: { id: string, children: React.ReactNode, className?: string }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });
  
  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    zIndex: isDragging ? 50 : 1,
  };

  // Parallax Effect State
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const rotateX = useTransform(y, [-100, 100], [5, -5]);
  const rotateY = useTransform(x, [-100, 100], [-5, 5]);

  function handleMouseMove(event: React.MouseEvent<HTMLDivElement, MouseEvent>) {
    const rect = event.currentTarget.getBoundingClientRect();
    x.set(event.clientX - rect.left - rect.width / 2);
    y.set(event.clientY - rect.top - rect.height / 2);
  }

  function handleMouseLeave() {
    x.set(0);
    y.set(0);
  }

  return (
    <motion.div
      ref={setNodeRef}
      style={{ ...style, rotateX, rotateY, perspective: 1000 }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className={`relative rounded-3xl p-7 shadow-xl backdrop-blur-xl border border-gray-200 dark:border-[#00FF41]/30 bg-white/80 dark:bg-[#0D1117]/80 hover:border-[#00FF41]/50 transition-colors duration-500 group overflow-hidden ${className}`}
    >
      {/* Holographic Glare */}
      <motion.div 
        className="absolute inset-0 z-0 pointer-events-none opacity-0 group-hover:opacity-20 transition-opacity duration-300"
        style={{
          background: "radial-gradient(circle at 50% 50%, #00FF41 0%, transparent 60%)",
          x: useTransform(x, [-100, 100], [-50, 50]),
          y: useTransform(y, [-100, 100], [-50, 50]),
        }}
      />
      
      {/* Drag Handle */}
      <div 
        {...attributes} 
        {...listeners} 
        className="absolute top-4 right-4 text-gray-400 hover:text-[#00FF41] cursor-grab active:cursor-grabbing z-20"
      >
        <GripHorizontal className="w-5 h-5" />
      </div>

      <div className="relative z-10 h-full">{children}</div>
    </motion.div>
  );
}


export default function Dashboard() {
  const [marketMode, setMarketMode] = useState<"TRADFI" | "CRYPTO">("TRADFI");
  const [ticker, setTicker] = useState("AAPL");
  const [priceData, setPriceData] = useState<{ time: string; price: number }[]>([]);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [wsStatus, setWsStatus] = useState("Connecting...");
  
  // Committee State
  const [committeeTranscript, setCommitteeTranscript] = useState<{agent: string, message: string}[] | null>(null);
  const [loadingCommittee, setLoadingCommittee] = useState(false);

  // Sound Effects
  const [playClick] = useSound(BEEP_SOUND, { volume: 0.25 });
  const [playTrade] = useSound(CASH_SOUND, { volume: 0.5 });

  // Drag and Drop Layout State
  const [layout, setLayout] = useState(["chart", "trading", "committee"]);
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );

  const handleDragEnd = (event: any) => {
    const { active, over } = event;
    if (active.id !== over.id) {
      setLayout((items) => {
        const oldIndex = items.indexOf(active.id);
        const newIndex = items.indexOf(over.id);
        return arrayMove(items, oldIndex, newIndex);
      });
    }
  };

  // WebSocket for Live Pricing
  useEffect(() => {
    setPriceData([]);
    setWsStatus("Connecting...");
    
    const ws = new WebSocket(`ws://localhost:8000/ws/price/${ticker}`);

    ws.onopen = () => setWsStatus("Connected");
    ws.onclose = () => setWsStatus("Disconnected");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setCurrentPrice(data.price);
      setPriceData((prev) => {
        const newData = [...prev, { time: new Date(data.timestamp).toLocaleTimeString(), price: data.price }];
        return newData.slice(-50);
      });
    };

    return () => ws.close();
  }, [ticker]);

  // Fetch Prediction
  const generatePrediction = async () => {
    playClick();
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, horizon_days: 7, include_sentiment: true, model_type: "ensemble" })
      });
      if (res.ok) {
        const data = await res.json();
        setPrediction(data);
      }
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  const consultCommittee = async () => {
    playClick();
    setLoadingCommittee(true);
    setCommitteeTranscript(null);
    try {
      const res = await fetch("http://localhost:8000/committee/debate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker })
      });
      if (res.ok) {
        const data = await res.json();
        setCommitteeTranscript(data.transcript);
      }
    } catch (err) {
      console.error(err);
    }
    setLoadingCommittee(false);
  };

  const triggerTrade = (side: string) => {
    playTrade();
    confetti({
      particleCount: 100,
      spread: 70,
      origin: { y: 0.6 },
      colors: ['#00FF41', '#ffffff']
    });
    // In production, this would hit /trading/order
    alert(`Mock ${side} Trade Executed for ${ticker}!`);
  };

  const handleVoiceIntent = (intent: any) => {
    playClick();
    if (intent.action === "switch_market") {
      if (intent.market === "CRYPTO") {
        setMarketMode("CRYPTO");
        setTicker(intent.ticker || "BTC/USD");
      } else {
        setMarketMode("TRADFI");
        setTicker(intent.ticker || "AAPL");
      }
    } else if (intent.action === "buy" || intent.action === "sell") {
      if (intent.ticker) setTicker(intent.ticker.toUpperCase());
      alert(`Voice Command Recognized: Ready to ${intent.action.toUpperCase()} ${intent.qty || 'market qty'} of ${intent.ticker || ticker}. Waiting for your confirmation.`);
    } else if (intent.action === "view_ticker") {
      if (intent.ticker) setTicker(intent.ticker.toUpperCase());
    }
  };

  const testNotification = async () => {
    playClick();
    try {
      const res = await fetch("http://localhost:8000/trading/test-alert", { method: "POST" });
      if (res.ok) {
        alert("Test notification triggered! Check your Discord.");
      } else {
        alert("Failed to send notification. Ensure DISCORD_WEBHOOK_URL is set in backend .env");
      }
    } catch (err) {
      console.error(err);
      alert("Error triggering test notification");
    }
  };

  return (
    <div className="min-h-screen bg-[#FAFAFA] dark:bg-[#0D1117] text-gray-900 dark:text-[#E8E0D0] font-sans selection:bg-[#00FF41] selection:text-black relative overflow-hidden transition-colors duration-1000">
      {/* Voice Copilot FAB */}
      <VoiceCopilot onIntentParsed={handleVoiceIntent} />

      {/* 3D Background Candlestick City */}
      <Scene />

      {/* Top Navbar */}
      <motion.nav 
        initial={{ y: -50, opacity: 0 }} 
        animate={{ y: 0, opacity: 1 }} 
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="border-b border-gray-200 dark:border-[#00FF41]/20 bg-white/70 dark:bg-[#0D1117]/70 backdrop-blur-xl sticky top-0 z-50 shadow-sm"
      >
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-black dark:text-[#00FF41] drop-shadow-[0_0_8px_rgba(0,255,65,0.6)]" />
            <span className="font-serif text-xl font-bold dark:text-white tracking-wide">StockSense Matrix</span>
            <span className="ml-4 px-2.5 py-0.5 rounded-full bg-gray-100 dark:bg-[#00FF41]/10 border border-gray-300 dark:border-[#00FF41]/40 text-black dark:text-[#00FF41] text-xs font-bold tracking-widest uppercase shadow-[0_0_10px_rgba(0,255,65,0.2)]">
              God Tier
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center bg-gray-100 dark:bg-[#111520] border border-gray-300 dark:border-[#00FF41]/20 rounded-lg p-1">
              <button 
                onClick={() => { playClick(); setMarketMode("TRADFI"); setTicker("AAPL"); }}
                className={`px-3 py-1 text-sm font-semibold rounded-md flex items-center gap-2 transition-all ${marketMode === "TRADFI" ? "bg-white dark:bg-[#00FF41]/20 text-black dark:text-[#00FF41] shadow-sm" : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"}`}
              >
                <Briefcase className="w-4 h-4" /> Stocks
              </button>
              <button 
                onClick={() => { playClick(); setMarketMode("CRYPTO"); setTicker("BTC/USD"); }}
                className={`px-3 py-1 text-sm font-semibold rounded-md flex items-center gap-2 transition-all ${marketMode === "CRYPTO" ? "bg-white dark:bg-[#00FF41]/20 text-black dark:text-[#00FF41] shadow-sm" : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"}`}
              >
                <Bitcoin className="w-4 h-4" /> Crypto
              </button>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 font-mono font-bold">
              <span className={`w-2 h-2 rounded-full shadow-[0_0_8px_currentColor] ${wsStatus === 'Connected' ? 'bg-[#00FF41] text-[#00FF41] animate-pulse' : 'bg-red-500 text-red-500'}`}></span>
              Live: {wsStatus}
            </div>
            <input 
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="bg-gray-100/80 dark:bg-[#080B0F]/80 backdrop-blur-sm border border-gray-300 dark:border-[#00FF41]/40 text-black dark:text-white rounded-lg px-4 py-1.5 focus:outline-none focus:border-[#00FF41] focus:shadow-[0_0_15px_rgba(0,255,65,0.3)] transition-all w-32 font-mono font-bold uppercase"
              placeholder={marketMode === "TRADFI" ? "AAPL" : "BTC/USD"}
            />
          </div>
        </div>
      </motion.nav>

      <main className="max-w-7xl mx-auto px-6 py-8 relative z-10">
        
        <motion.div variants={containerVariants} initial="hidden" animate="show">
          
          {/* Hero Section */}
          <motion.div variants={itemVariants} className="relative overflow-hidden rounded-3xl bg-white/80 dark:bg-[#111520]/80 backdrop-blur-xl border border-gray-200 dark:border-[#00FF41]/30 p-8 mb-8 shadow-2xl dark:shadow-[0_0_40px_rgba(0,255,65,0.05)] group">
            <div className="absolute top-0 right-0 w-96 h-96 bg-radial-gradient from-gray-200 dark:from-[#00FF41]/20 to-transparent -translate-y-1/2 translate-x-1/2 rounded-full blur-3xl opacity-50 group-hover:opacity-80 transition-opacity duration-700"></div>
            
            <div className="flex justify-between items-start relative z-10">
              <div>
                <motion.h1 
                  key={ticker}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="text-6xl font-serif font-bold text-black dark:text-white mb-3 drop-shadow-md dark:drop-shadow-[0_2px_10px_rgba(0,255,65,0.3)]"
                >
                  {ticker}
                </motion.h1>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-6 tracking-wide font-mono font-bold">God Tier Data Analysis</p>
                <div className="flex gap-3">
                  <span className="px-4 py-1.5 rounded-full text-xs font-bold bg-gray-100 dark:bg-[#00FF41]/10 border border-gray-300 dark:border-[#00FF41]/40 text-gray-800 dark:text-[#00FF41] backdrop-blur-md shadow-sm">LSTM + XGBoost</span>
                  <span className="px-4 py-1.5 rounded-full text-xs font-bold bg-gray-100 dark:bg-[#00FF41]/10 border border-gray-300 dark:border-[#00FF41]/40 text-gray-800 dark:text-[#00FF41] backdrop-blur-md shadow-sm">Alpaca Executions</span>
                  <Link href="/backtest" onClick={() => playClick()} className="px-4 py-1.5 rounded-full text-xs font-bold bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 text-gray-800 dark:text-white/80 hover:bg-black/10 dark:hover:bg-white/10 transition-all backdrop-blur-md flex items-center gap-2">
                    <History className="w-3 h-3" /> Time Machine
                  </Link>
                </div>
              </div>
              <div className="text-right mt-2">
                <motion.div 
                  key={currentPrice}
                  initial={{ scale: 0.95 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 300, damping: 20 }}
                  className="text-5xl font-mono font-black text-black dark:text-white drop-shadow-md dark:drop-shadow-[0_0_15px_rgba(0,255,65,0.4)]"
                >
                  ${currentPrice ? currentPrice.toFixed(2) : "---"}
                </motion.div>
              </div>
            </div>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Draggable Main Column */}
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
              <div className="lg:col-span-2 space-y-8 flex flex-col">
                <SortableContext items={layout} strategy={verticalListSortingStrategy}>
                  
                  {layout.map((id) => (
                    <div key={id}>
                      {id === "chart" && (
                        <SortableHolographicCard id="chart">
                          <h2 className="text-xl font-serif font-bold text-gray-800 dark:text-[#00FF41] mb-6 flex items-center gap-3 drop-shadow-sm dark:drop-shadow-[0_0_8px_rgba(0,255,65,0.5)]">
                            <TrendingUp className="w-5 h-5" /> Live Tick Chart
                          </h2>
                          <div className="h-[350px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={priceData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(150,150,150,0.1)" vertical={false} />
                                <XAxis dataKey="time" stroke="#888" fontSize={11} tickMargin={10} axisLine={false} tickLine={false} />
                                <YAxis domain={['auto', 'auto']} stroke="#888" fontSize={11} width={60} tickFormatter={(v) => `$${v.toFixed(2)}`} axisLine={false} tickLine={false} />
                                <Tooltip 
                                  contentStyle={{ backgroundColor: 'rgba(255,255,255,0.9)', backdropFilter: 'blur(10px)', borderRadius: '12px', borderColor: '#ccc', color: '#000', boxShadow: '0 10px 30px -10px rgba(0,0,0,0.1)' }}
                                  itemStyle={{ color: '#000', fontWeight: 'bold' }}
                                />
                                <Line 
                                  type="monotone" 
                                  dataKey="price" 
                                  stroke="#00FF41" 
                                  strokeWidth={4} 
                                  dot={false} 
                                  isAnimationActive={false} 
                                  style={{ filter: "drop-shadow(0px 4px 6px rgba(0, 255, 65, 0.4))" }}
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </SortableHolographicCard>
                      )}

                      {id === "trading" && (
                        <SortableHolographicCard id="trading">
                           <h2 className="text-xl font-serif font-bold text-gray-800 dark:text-[#00FF41] mb-4 flex items-center gap-3 drop-shadow-sm dark:drop-shadow-[0_0_8px_rgba(0,255,65,0.5)]">
                            <DollarSign className="w-5 h-5" /> Gamified Execution
                          </h2>
                          <p className="text-sm text-gray-500 dark:text-gray-400 mb-6 font-mono font-bold">Signals generated by the ML ensemble can automatically trigger real trades.</p>
                          <div className="flex gap-5">
                            <motion.button onClick={() => triggerTrade("Long")} whileHover={{ scale: 1.02, backgroundColor: "rgba(0,255,65,0.2)" }} whileTap={{ scale: 0.98 }} className="flex-1 bg-[#00FF41]/10 border border-[#00FF41]/40 text-[#00FF41] font-bold py-4 rounded-2xl transition-all flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(0,255,65,0.2)]">
                              <ArrowUpRight className="w-5 h-5" /> Go Long (Market)
                            </motion.button>
                            <motion.button onClick={() => triggerTrade("Short")} whileHover={{ scale: 1.02, backgroundColor: "rgba(248,113,113,0.2)" }} whileTap={{ scale: 0.98 }} className="flex-1 bg-[#F87171]/10 border border-[#F87171]/40 text-[#F87171] font-bold py-4 rounded-2xl transition-all flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(248,113,113,0.2)]">
                              <ArrowDownRight className="w-5 h-5" /> Go Short (Market)
                            </motion.button>
                          </div>
                          <div className="mt-4">
                            <motion.button onClick={testNotification} whileHover={{ scale: 1.02, backgroundColor: "rgba(100,100,255,0.2)" }} whileTap={{ scale: 0.98 }} className="w-full bg-[#5865F2]/10 border border-[#5865F2]/40 text-[#5865F2] font-bold py-3 rounded-2xl transition-all flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(88,101,242,0.2)]">
                              <BellRing className="w-4 h-4" /> Test Discord Webhook Alert
                            </motion.button>
                          </div>
                        </SortableHolographicCard>
                      )}

                      {id === "committee" && (
                        <SortableHolographicCard id="committee">
                           <h2 className="text-xl font-serif font-bold text-gray-800 dark:text-[#00FF41] mb-4 flex items-center gap-3 drop-shadow-sm dark:drop-shadow-[0_0_8px_rgba(0,255,65,0.5)]">
                            <Users className="w-5 h-5" /> AI Investment Committee
                          </h2>
                          <p className="text-sm text-gray-500 dark:text-gray-400 mb-6 font-mono font-bold">Consult a swarm of 4 specialized AI agents to debate this asset before trading.</p>
                          
                          <motion.button 
                            whileHover={{ scale: 1.01, backgroundColor: "rgba(0,255,65,0.15)" }} 
                            whileTap={{ scale: 0.99 }} 
                            onClick={consultCommittee}
                            disabled={loadingCommittee}
                            className="w-full bg-black/5 dark:bg-[#00FF41]/10 border border-gray-300 dark:border-[#00FF41]/40 text-black dark:text-white font-bold py-4 rounded-2xl transition-all flex items-center justify-center gap-2 shadow-sm dark:shadow-[0_0_15px_rgba(0,255,65,0.1)] disabled:opacity-50"
                          >
                            {loadingCommittee ? <Activity className="w-5 h-5 animate-spin" /> : <MessageSquare className="w-5 h-5" />}
                            {loadingCommittee ? "Agents are debating..." : "Consult Committee"}
                          </motion.button>

                          <AnimatePresence>
                            {committeeTranscript && (
                              <motion.div 
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: "auto" }}
                                className="mt-6 space-y-4"
                              >
                                {committeeTranscript.map((msg, idx) => (
                                  <motion.div 
                                    key={idx}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: idx * 0.2 }}
                                    className="p-4 rounded-2xl border border-gray-200 dark:border-[#00FF41]/20 bg-gray-50/80 dark:bg-black/40 backdrop-blur-md flex gap-4 items-start shadow-sm"
                                  >
                                    <div className="w-10 h-10 rounded-full bg-gray-200 dark:bg-[#00FF41]/20 flex items-center justify-center border border-gray-300 dark:border-[#00FF41]/40 shrink-0">
                                      <span className="font-black text-black dark:text-white text-xs">{msg.agent[0]}</span>
                                    </div>
                                    <div>
                                      <div className="text-xs text-gray-500 dark:text-[#00FF41] uppercase tracking-widest font-black mb-1">{msg.agent}</div>
                                      <div className="text-sm text-gray-800 dark:text-gray-300 leading-relaxed font-mono">{msg.message}</div>
                                    </div>
                                  </motion.div>
                                ))}
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </SortableHolographicCard>
                      )}
                    </div>
                  ))}
                </SortableContext>
              </div>
            </DndContext>

            {/* Sidebar / Prediction (Fixed) */}
            <motion.div variants={itemVariants} className="space-y-8">
              <div className="bg-white/80 dark:bg-[#0E1118]/80 backdrop-blur-xl border border-gray-200 dark:border-[#00FF41]/20 rounded-3xl p-7 shadow-2xl relative overflow-hidden group hover:border-[#00FF41]/50 transition-colors duration-500">
                <div className="absolute top-0 right-0 w-32 h-32 bg-gray-200 dark:bg-[#00FF41]/10 rounded-full blur-3xl group-hover:bg-gray-300 dark:group-hover:bg-[#00FF41]/20 transition-colors duration-500"></div>
                <h2 className="text-xl font-serif font-bold text-gray-800 dark:text-[#00FF41] mb-6 flex items-center gap-3 drop-shadow-sm dark:drop-shadow-[0_0_8px_rgba(0,255,65,0.5)] relative z-10">
                  <Crosshair className="w-5 h-5" /> AI Prediction
                </h2>
                
                <motion.button 
                  whileHover={{ scale: 1.02, boxShadow: "0 0 20px rgba(0,255,65,0.3)" }}
                  whileTap={{ scale: 0.98 }}
                  onClick={generatePrediction}
                  disabled={loading}
                  className="w-full bg-black dark:bg-gradient-to-r dark:from-[#00FF41]/20 dark:to-[#00FF41]/5 border border-black dark:border-[#00FF41]/50 text-white font-bold py-4 rounded-2xl transition-all flex items-center justify-center gap-3 disabled:opacity-50 relative z-10 shadow-lg"
                >
                  {loading ? (
                    <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: "linear" }}>
                      <Activity className="w-5 h-5" />
                    </motion.div>
                  ) : "Generate 7-Day Forecast"} 
                  {!loading && <ArrowRight className="w-5 h-5" />}
                </motion.button>

                {prediction && (
                  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mt-8 space-y-5 relative z-10">
                    <div className="p-5 rounded-2xl border border-gray-200 dark:border-[#00FF41]/30 bg-gray-50/80 dark:bg-black/40 backdrop-blur-md shadow-inner dark:shadow-[inset_0_0_20px_rgba(0,255,65,0.05)]">
                      <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-widest font-black mb-2">Target Price</div>
                      <div className="text-4xl font-mono font-black text-black dark:text-white drop-shadow-sm dark:drop-shadow-[0_0_10px_rgba(0,255,65,0.3)]">${prediction.predicted_price.toFixed(2)}</div>
                      <div className={`text-sm mt-2 font-mono font-black ${prediction.predicted_return >= 0 ? 'text-[#00FF41] drop-shadow-[0_0_5px_rgba(0,255,65,0.5)]' : 'text-red-500 drop-shadow-[0_0_5px_rgba(248,113,113,0.5)]'}`}>
                        {prediction.predicted_return >= 0 ? '+' : ''}{prediction.predicted_return.toFixed(2)}%
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-5">
                      <div className="p-5 rounded-2xl border border-gray-200 dark:border-[#00FF41]/20 bg-gray-50/80 dark:bg-black/40 backdrop-blur-md shadow-sm">
                        <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-widest font-black mb-2">Signal</div>
                        <div className={`text-xl font-black tracking-widest ${
                          prediction.signal === 'BUY' ? 'text-[#00FF41] drop-shadow-[0_0_8px_rgba(0,255,65,0.5)]' : 
                          prediction.signal === 'SELL' ? 'text-red-500 drop-shadow-[0_0_8px_rgba(248,113,113,0.5)]' : 'text-gray-800 dark:text-gray-300'
                        }`}>
                          {prediction.signal}
                        </div>
                      </div>
                      <div className="p-5 rounded-2xl border border-gray-200 dark:border-[#00FF41]/20 bg-gray-50/80 dark:bg-black/40 backdrop-blur-md shadow-sm">
                        <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-widest font-black mb-2">Confidence</div>
                        <div className="text-xl font-mono font-black text-black dark:text-white">{(prediction.confidence * 100).toFixed(1)}%</div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            </motion.div>

          </div>
        </motion.div>
      </main>
    </div>
  );
}

