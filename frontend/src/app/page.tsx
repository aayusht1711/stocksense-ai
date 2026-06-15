"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { motion, Variants, AnimatePresence } from "framer-motion";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Activity, TrendingUp, DollarSign, Crosshair, ArrowRight, ArrowUpRight, ArrowDownRight, Bitcoin, Briefcase, Users, MessageSquare, History, BellRing } from "lucide-react";
import Link from "next/link";
import VoiceCopilot from "../components/VoiceCopilot";

// Dynamically import the 3D scene to prevent SSR issues with WebGL
const Scene = dynamic(() => import("../components/Scene"), { ssr: false });

// Framer Motion Animation Variants
const containerVariants: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.15, delayChildren: 0.2 }
  }
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 30 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 70, damping: 15 } }
};

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

  const handleVoiceIntent = (intent: any) => {
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
    <div className="min-h-screen bg-[#080B0F] text-[#E8E0D0] font-sans selection:bg-[#C4A050] selection:text-black relative overflow-hidden">
      {/* Voice Copilot FAB */}
      <VoiceCopilot onIntentParsed={handleVoiceIntent} />

      {/* 3D Background */}
      <Scene />

      {/* Top Navbar */}
      <motion.nav 
        initial={{ y: -50, opacity: 0 }} 
        animate={{ y: 0, opacity: 1 }} 
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="border-b border-[#C4A050]/20 bg-[#0C0F14]/70 backdrop-blur-xl sticky top-0 z-50 shadow-[0_0_30px_rgba(196,160,80,0.05)]"
      >
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-[#F0D888] drop-shadow-[0_0_8px_rgba(240,216,136,0.6)]" />
            <span className="font-serif text-xl text-[#F0D888] tracking-wide">StockSense AI</span>
            <span className="ml-4 px-2.5 py-0.5 rounded-full bg-[#C4A050]/10 border border-[#C4A050]/40 text-[#C4A050] text-xs font-medium tracking-widest uppercase shadow-[0_0_10px_rgba(196,160,80,0.2)]">
              Pro
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center bg-[#111520] border border-[#C4A050]/20 rounded-lg p-1">
              <button 
                onClick={() => { setMarketMode("TRADFI"); setTicker("AAPL"); }}
                className={`px-3 py-1 text-sm font-semibold rounded-md flex items-center gap-2 transition-all ${marketMode === "TRADFI" ? "bg-[#C4A050]/20 text-[#F0D888] shadow-sm" : "text-gray-500 hover:text-gray-300"}`}
              >
                <Briefcase className="w-4 h-4" /> Stocks
              </button>
              <button 
                onClick={() => { setMarketMode("CRYPTO"); setTicker("BTC/USD"); }}
                className={`px-3 py-1 text-sm font-semibold rounded-md flex items-center gap-2 transition-all ${marketMode === "CRYPTO" ? "bg-[#C4A050]/20 text-[#F0D888] shadow-sm" : "text-gray-500 hover:text-gray-300"}`}
              >
                <Bitcoin className="w-4 h-4" /> Crypto
              </button>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-400 font-mono">
              <span className={`w-2 h-2 rounded-full shadow-[0_0_8px_currentColor] ${wsStatus === 'Connected' ? 'bg-green-500 text-green-500 animate-pulse' : 'bg-red-500 text-red-500'}`}></span>
              Live: {wsStatus}
            </div>
            <input 
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="bg-[#080B0F]/80 backdrop-blur-sm border border-[#C4A050]/40 text-white rounded-lg px-4 py-1.5 focus:outline-none focus:border-[#F0D888] focus:shadow-[0_0_15px_rgba(240,216,136,0.3)] transition-all w-32 font-mono"
              placeholder={marketMode === "TRADFI" ? "AAPL" : "BTC/USD"}
            />
          </div>
        </div>
      </motion.nav>

      <main className="max-w-7xl mx-auto px-6 py-8 relative z-10">
        
        <motion.div variants={containerVariants} initial="hidden" animate="show">
          
          {/* Hero Section */}
          <motion.div variants={itemVariants} className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-[#0C0F14]/80 to-[#111520]/80 backdrop-blur-xl border border-[#C4A050]/30 p-8 mb-8 shadow-[0_0_40px_rgba(196,160,80,0.05)] group">
            <div className="absolute top-0 right-0 w-96 h-96 bg-radial-gradient from-[#C4A050]/20 to-transparent -translate-y-1/2 translate-x-1/2 rounded-full blur-3xl opacity-50 group-hover:opacity-80 transition-opacity duration-700"></div>
            
            <div className="flex justify-between items-start relative z-10">
              <div>
                <motion.h1 
                  key={ticker}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="text-6xl font-serif text-[#F0D888] mb-3 drop-shadow-[0_2px_10px_rgba(240,216,136,0.3)]"
                >
                  {ticker}
                </motion.h1>
                <p className="text-sm text-gray-400 mb-6 tracking-wide font-mono">Real-time Advanced ML Analysis</p>
                <div className="flex gap-3">
                  <span className="px-4 py-1.5 rounded-full text-xs font-semibold bg-[#C4A050]/10 border border-[#C4A050]/40 text-[#C4A050] backdrop-blur-md shadow-[0_0_15px_rgba(196,160,80,0.1)]">LSTM + XGBoost</span>
                  <span className="px-4 py-1.5 rounded-full text-xs font-semibold bg-[#C4A050]/10 border border-[#C4A050]/40 text-[#C4A050] backdrop-blur-md shadow-[0_0_15px_rgba(196,160,80,0.1)]">Alpaca Executions</span>
                  <Link href="/backtest" className="px-4 py-1.5 rounded-full text-xs font-semibold bg-white/5 border border-white/10 text-white/80 hover:bg-white/10 hover:text-white transition-all backdrop-blur-md flex items-center gap-2">
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
                  className="text-5xl font-mono text-[#F0D888] drop-shadow-[0_0_15px_rgba(240,216,136,0.4)]"
                >
                  ${currentPrice ? currentPrice.toFixed(2) : "---"}
                </motion.div>
              </div>
            </div>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Main Chart */}
            <motion.div variants={itemVariants} className="lg:col-span-2 space-y-8">
              <div className="bg-[#0E1118]/70 backdrop-blur-xl border border-[#C4A050]/20 rounded-3xl p-7 hover:border-[#C4A050]/50 transition-colors duration-500 shadow-xl group">
                <h2 className="text-xl font-serif text-[#C4A050] mb-6 flex items-center gap-3 drop-shadow-[0_0_8px_rgba(196,160,80,0.5)]">
                  <TrendingUp className="w-5 h-5" /> Live Tick Chart
                </h2>
                <div className="h-[350px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={priceData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
                      <XAxis dataKey="time" stroke="#555" fontSize={11} tickMargin={10} axisLine={false} tickLine={false} />
                      <YAxis domain={['auto', 'auto']} stroke="#555" fontSize={11} width={60} tickFormatter={(v) => `$${v.toFixed(2)}`} axisLine={false} tickLine={false} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: 'rgba(12,15,20,0.9)', backdropFilter: 'blur(10px)', borderRadius: '12px', borderColor: 'rgba(196,160,80,0.4)', color: '#E8E0D0', boxShadow: '0 10px 30px -10px rgba(0,0,0,0.5)' }}
                        itemStyle={{ color: '#F0D888', fontWeight: 'bold' }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="price" 
                        stroke="#4ADE80" 
                        strokeWidth={3} 
                        dot={false} 
                        isAnimationActive={false} 
                        style={{ filter: "drop-shadow(0px 4px 6px rgba(74, 222, 128, 0.4))" }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Trading Panel */}
              <motion.div variants={itemVariants} className="bg-[#0E1118]/70 backdrop-blur-xl border border-[#C4A050]/20 rounded-3xl p-7 shadow-xl">
                 <h2 className="text-xl font-serif text-[#C4A050] mb-4 flex items-center gap-3 drop-shadow-[0_0_8px_rgba(196,160,80,0.5)]">
                  <DollarSign className="w-5 h-5" /> Automated Execution
                </h2>
                <p className="text-sm text-gray-400 mb-6 font-mono">Signals generated by the ML ensemble can automatically trigger real trades via the Alpaca API.</p>
                <div className="flex gap-5">
                  <motion.button whileHover={{ scale: 1.02, backgroundColor: "rgba(74,222,128,0.2)" }} whileTap={{ scale: 0.98 }} className="flex-1 bg-[#4ADE80]/10 border border-[#4ADE80]/40 text-[#4ADE80] font-semibold py-4 rounded-2xl transition-all flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(74,222,128,0.1)]">
                    <ArrowUpRight className="w-5 h-5" /> Go Long (Market)
                  </motion.button>
                  <motion.button whileHover={{ scale: 1.02, backgroundColor: "rgba(248,113,113,0.2)" }} whileTap={{ scale: 0.98 }} className="flex-1 bg-[#F87171]/10 border border-[#F87171]/40 text-[#F87171] font-semibold py-4 rounded-2xl transition-all flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(248,113,113,0.1)]">
                    <ArrowDownRight className="w-5 h-5" /> Go Short (Market)
                  </motion.button>
                </div>
                <div className="mt-4">
                  <motion.button onClick={testNotification} whileHover={{ scale: 1.02, backgroundColor: "rgba(100,100,255,0.2)" }} whileTap={{ scale: 0.98 }} className="w-full bg-[#5865F2]/10 border border-[#5865F2]/40 text-[#5865F2] font-semibold py-3 rounded-2xl transition-all flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(88,101,242,0.1)]">
                    <BellRing className="w-4 h-4" /> Test Discord Webhook Alert
                  </motion.button>
                </div>
              </motion.div>

              {/* AI Committee Panel */}
              <motion.div variants={itemVariants} className="bg-[#0E1118]/70 backdrop-blur-xl border border-[#C4A050]/20 rounded-3xl p-7 shadow-xl mt-8">
                 <h2 className="text-xl font-serif text-[#C4A050] mb-4 flex items-center gap-3 drop-shadow-[0_0_8px_rgba(196,160,80,0.5)]">
                  <Users className="w-5 h-5" /> AI Investment Committee
                </h2>
                <p className="text-sm text-gray-400 mb-6 font-mono">Consult a swarm of 4 specialized AI agents (Quant, Fundamental, Sentiment, Risk Manager) to debate this asset before trading.</p>
                
                <motion.button 
                  whileHover={{ scale: 1.01, backgroundColor: "rgba(196,160,80,0.15)" }} 
                  whileTap={{ scale: 0.99 }} 
                  onClick={consultCommittee}
                  disabled={loadingCommittee}
                  className="w-full bg-[#C4A050]/10 border border-[#C4A050]/40 text-[#F0D888] font-semibold py-4 rounded-2xl transition-all flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(196,160,80,0.1)] disabled:opacity-50"
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
                          className="p-4 rounded-2xl border border-[#C4A050]/20 bg-black/40 backdrop-blur-md flex gap-4 items-start"
                        >
                          <div className="w-10 h-10 rounded-full bg-[#C4A050]/20 flex items-center justify-center border border-[#C4A050]/40 shrink-0">
                            <span className="font-bold text-[#F0D888] text-xs">{msg.agent[0]}</span>
                          </div>
                          <div>
                            <div className="text-xs text-[#C4A050] uppercase tracking-widest font-semibold mb-1">{msg.agent}</div>
                            <div className="text-sm text-gray-300 leading-relaxed">{msg.message}</div>
                          </div>
                        </motion.div>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            </motion.div>

            {/* Sidebar / Prediction */}
            <motion.div variants={itemVariants} className="space-y-8">
              <div className="bg-[#0E1118]/80 backdrop-blur-xl border border-[#C4A050]/20 rounded-3xl p-7 shadow-xl relative overflow-hidden group">
                <div className="absolute top-0 right-0 w-32 h-32 bg-[#C4A050]/10 rounded-full blur-3xl group-hover:bg-[#C4A050]/20 transition-colors duration-500"></div>
                <h2 className="text-xl font-serif text-[#C4A050] mb-6 flex items-center gap-3 drop-shadow-[0_0_8px_rgba(196,160,80,0.5)] relative z-10">
                  <Crosshair className="w-5 h-5" /> AI Prediction
                </h2>
                
                <motion.button 
                  whileHover={{ scale: 1.02, boxShadow: "0 0 20px rgba(196,160,80,0.3)" }}
                  whileTap={{ scale: 0.98 }}
                  onClick={generatePrediction}
                  disabled={loading}
                  className="w-full bg-gradient-to-r from-[#C4A050]/20 to-[#C4A050]/5 border border-[#C4A050]/50 text-[#F0D888] font-semibold py-4 rounded-2xl transition-all flex items-center justify-center gap-3 disabled:opacity-50 relative z-10 shadow-lg"
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
                    <div className="p-5 rounded-2xl border border-[#C4A050]/30 bg-black/40 backdrop-blur-md shadow-[inset_0_0_20px_rgba(196,160,80,0.05)]">
                      <div className="text-xs text-gray-400 uppercase tracking-widest font-semibold mb-2">Target Price</div>
                      <div className="text-4xl font-mono text-[#F0D888] drop-shadow-[0_0_10px_rgba(240,216,136,0.3)]">${prediction.predicted_price.toFixed(2)}</div>
                      <div className={`text-sm mt-2 font-mono font-bold ${prediction.predicted_return >= 0 ? 'text-green-400 drop-shadow-[0_0_5px_rgba(74,222,128,0.5)]' : 'text-red-400 drop-shadow-[0_0_5px_rgba(248,113,113,0.5)]'}`}>
                        {prediction.predicted_return >= 0 ? '+' : ''}{prediction.predicted_return.toFixed(2)}%
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-5">
                      <div className="p-5 rounded-2xl border border-[#C4A050]/20 bg-black/40 backdrop-blur-md">
                        <div className="text-xs text-gray-400 uppercase tracking-widest font-semibold mb-2">Signal</div>
                        <div className={`text-xl font-black tracking-widest ${
                          prediction.signal === 'BUY' ? 'text-green-400 drop-shadow-[0_0_8px_rgba(74,222,128,0.5)]' : 
                          prediction.signal === 'SELL' ? 'text-red-400 drop-shadow-[0_0_8px_rgba(248,113,113,0.5)]' : 'text-[#C4A050]'
                        }`}>
                          {prediction.signal}
                        </div>
                      </div>
                      <div className="p-5 rounded-2xl border border-[#C4A050]/20 bg-black/40 backdrop-blur-md">
                        <div className="text-xs text-gray-400 uppercase tracking-widest font-semibold mb-2">Confidence</div>
                        <div className="text-xl font-mono text-white">{(prediction.confidence * 100).toFixed(1)}%</div>
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
