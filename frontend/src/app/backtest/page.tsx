"use client";

import { useEffect, useState, useRef } from "react";
import dynamic from "next/dynamic";
import { motion, Variants } from "framer-motion";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Play, Square, Activity, History, ArrowLeft, TrendingUp, AlertTriangle } from "lucide-react";
import Link from "next/link";

// Dynamically import the 3D scene to prevent SSR issues with WebGL
const Scene = dynamic(() => import("../../components/Scene"), { ssr: false });

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

export default function Backtest() {
  const [ticker, setTicker] = useState("AAPL");
  const [isPlaying, setIsPlaying] = useState(false);
  const [historyData, setHistoryData] = useState<{ date: string; price: number, signal: string }[]>([]);
  const [currentDrawdown, setCurrentDrawdown] = useState(0);
  const [status, setStatus] = useState("Idle");
  const wsRef = useRef<WebSocket | null>(null);

  const startReplay = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    setHistoryData([]);
    setIsPlaying(true);
    setStatus("Buffering Time Machine...");

    const ws = new WebSocket(`ws://localhost:8000/ws/backtest/${ticker}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("Streaming History...");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status === "COMPLETE") {
        setIsPlaying(false);
        setStatus("Simulation Complete");
        return;
      }

      setHistoryData(prev => [...prev, data]);
      setCurrentDrawdown(data.drawdown || 0);
    };

    ws.onerror = (err) => {
      console.error(err);
      setStatus("Error Connecting");
      setIsPlaying(false);
    };

    ws.onclose = () => {
      if (status !== "Simulation Complete") {
        setStatus("Idle");
      }
      setIsPlaying(false);
    };
  };

  const stopReplay = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setIsPlaying(false);
    setStatus("Stopped");
  };

  // Determine atmospheric color based on drawdown (red during crashes)
  const isCrash = currentDrawdown < -15;

  return (
    <div className={`min-h-screen ${isCrash ? 'bg-[#1a0a0a]' : 'bg-[#080B0F]'} text-[#E8E0D0] font-sans selection:bg-[#C4A050] selection:text-black relative overflow-hidden transition-colors duration-1000`}>
      {/* 3D Background */}
      <div className="absolute inset-0 z-0 opacity-40 pointer-events-none">
        <Scene />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Navbar */}
        <motion.nav initial={{ y: -20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="flex justify-between items-center mb-12 bg-black/20 backdrop-blur-md p-4 rounded-2xl border border-white/5">
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-2 text-[#C4A050] hover:text-[#F0D888] transition-colors font-serif text-xl tracking-tight">
              <ArrowLeft className="w-5 h-5" /> StockSense AI
            </Link>
            <div className="h-6 w-px bg-white/10"></div>
            <div className="flex items-center gap-2 text-sm text-gray-400 font-mono">
              <History className="w-4 h-4" /> Time Machine
            </div>
          </div>
          <div className="flex items-center gap-4">
             <input 
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              disabled={isPlaying}
              className="bg-[#080B0F]/80 backdrop-blur-sm border border-[#C4A050]/40 text-white rounded-lg px-4 py-1.5 focus:outline-none focus:border-[#F0D888] disabled:opacity-50 transition-all w-32 font-mono"
              placeholder="TICKER"
            />
          </div>
        </motion.nav>

        <motion.div variants={containerVariants} initial="hidden" animate="visible">
          
          {/* Controls Section */}
          <motion.div variants={itemVariants} className={`relative overflow-hidden rounded-3xl backdrop-blur-xl border p-8 mb-8 shadow-2xl ${isCrash ? 'bg-[#2a0b0b]/80 border-red-500/30' : 'bg-[#0C0F14]/80 border-[#C4A050]/30'}`}>
            <div className="flex justify-between items-center relative z-10">
              <div>
                <h1 className="text-4xl font-serif text-[#F0D888] mb-2 flex items-center gap-3">
                  Historical Replay <span className="text-sm font-mono text-gray-400 font-normal tracking-widest uppercase">({status})</span>
                </h1>
                <p className="text-sm text-gray-400">Stream years of historical ticks and watch the ML engine react dynamically.</p>
              </div>
              <div className="flex gap-4">
                {!isPlaying ? (
                   <motion.button 
                    whileHover={{ scale: 1.05 }} 
                    whileTap={{ scale: 0.95 }} 
                    onClick={startReplay}
                    className="bg-[#C4A050] text-black font-bold py-3 px-8 rounded-xl shadow-[0_0_20px_rgba(196,160,80,0.4)] hover:shadow-[0_0_30px_rgba(240,216,136,0.6)] transition-all flex items-center gap-2"
                  >
                    <Play className="w-5 h-5 fill-black" /> Run Simulation
                  </motion.button>
                ) : (
                   <motion.button 
                    whileHover={{ scale: 1.05 }} 
                    whileTap={{ scale: 0.95 }} 
                    onClick={stopReplay}
                    className="bg-red-500/20 text-red-400 border border-red-500/50 font-bold py-3 px-8 rounded-xl hover:bg-red-500/30 transition-all flex items-center gap-2"
                  >
                    <Square className="w-5 h-5 fill-red-400" /> Stop
                  </motion.button>
                )}
              </div>
            </div>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Chart */}
            <motion.div variants={itemVariants} className="lg:col-span-2">
              <div className={`backdrop-blur-xl border rounded-3xl p-7 transition-colors duration-500 shadow-xl ${isCrash ? 'bg-[#1a0a0a]/70 border-red-500/20' : 'bg-[#0E1118]/70 border-[#C4A050]/20'}`}>
                <h2 className={`text-xl font-serif mb-6 flex items-center gap-3 ${isCrash ? 'text-red-400' : 'text-[#C4A050]'}`}>
                  <TrendingUp className="w-5 h-5" /> Timeline View
                </h2>
                <div className="h-[400px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={historyData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis dataKey="date" stroke="#555" fontSize={11} tickMargin={10} axisLine={false} tickLine={false} minTickGap={30} />
                      <YAxis domain={['auto', 'auto']} stroke="#555" fontSize={11} width={60} tickFormatter={(v) => `$${v.toFixed(0)}`} axisLine={false} tickLine={false} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: 'rgba(12,15,20,0.9)', backdropFilter: 'blur(10px)', borderRadius: '12px', borderColor: 'rgba(196,160,80,0.4)', color: '#E8E0D0' }}
                        itemStyle={{ color: '#F0D888', fontWeight: 'bold' }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="price" 
                        stroke={isCrash ? "#F87171" : "#F0D888"} 
                        strokeWidth={2} 
                        dot={false} 
                        isAnimationActive={false} 
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </motion.div>

            {/* AI Log */}
            <motion.div variants={itemVariants} className="h-full">
               <div className={`h-full backdrop-blur-xl border rounded-3xl p-7 shadow-xl flex flex-col ${isCrash ? 'bg-[#1a0a0a]/70 border-red-500/20' : 'bg-[#0E1118]/70 border-[#C4A050]/20'}`}>
                 <h2 className={`text-xl font-serif mb-6 flex items-center gap-3 ${isCrash ? 'text-red-400' : 'text-[#C4A050]'}`}>
                  <Activity className="w-5 h-5" /> AI Signal Log
                </h2>
                
                <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin scrollbar-thumb-[#C4A050]/30 scrollbar-track-transparent">
                  {historyData.slice().reverse().map((data, idx) => (
                    <div key={idx} className="p-3 rounded-xl bg-black/40 border border-white/5 flex justify-between items-center text-sm">
                      <div className="text-gray-400 font-mono">{data.date}</div>
                      <div className="flex items-center gap-3">
                        <span className="text-white font-mono">${data.price.toFixed(2)}</span>
                        <span className={`px-2 py-1 rounded text-xs font-bold ${data.signal === 'BUY' ? 'bg-green-500/20 text-green-400' : data.signal === 'SELL' ? 'bg-red-500/20 text-red-400' : 'bg-gray-500/20 text-gray-400'}`}>
                          {data.signal}
                        </span>
                      </div>
                    </div>
                  ))}
                  {historyData.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-gray-500 text-sm text-center">
                      <AlertTriangle className="w-8 h-8 mb-3 opacity-50" />
                      Hit Run Simulation to watch the AI decisions historically.
                    </div>
                  )}
                </div>
               </div>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
