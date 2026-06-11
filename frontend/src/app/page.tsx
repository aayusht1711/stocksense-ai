"use client";

import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Activity, TrendingUp, DollarSign, Crosshair, ArrowRight, ArrowUpRight, ArrowDownRight } from "lucide-react";

export default function Dashboard() {
  const [ticker, setTicker] = useState("AAPL");
  const [priceData, setPriceData] = useState<{ time: string; price: number }[]>([]);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [wsStatus, setWsStatus] = useState("Connecting...");

  // WebSocket for Live Pricing
  useEffect(() => {
    setPriceData([]);
    setWsStatus("Connecting...");
    
    // In development, Next runs on 3000, FastAPI on 8000
    const ws = new WebSocket(`ws://localhost:8000/ws/price/${ticker}`);

    ws.onopen = () => setWsStatus("Connected");
    ws.onclose = () => setWsStatus("Disconnected");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setCurrentPrice(data.price);
      setPriceData((prev) => {
        const newData = [...prev, { time: new Date(data.timestamp).toLocaleTimeString(), price: data.price }];
        return newData.slice(-50); // Keep last 50 ticks
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

  return (
    <div className="min-h-screen bg-[#080B0F] text-[#E8E0D0] font-sans selection:bg-[#C4A050] selection:text-black">
      {/* Top Navbar */}
      <nav className="border-b border-[#C4A050]/20 bg-[#0C0F14]/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-[#F0D888]" />
            <span className="font-serif text-xl text-[#F0D888] tracking-wide">StockSense AI</span>
            <span className="ml-4 px-2.5 py-0.5 rounded-full bg-[#C4A050]/10 border border-[#C4A050]/20 text-[#C4A050] text-xs font-medium tracking-widest uppercase">
              Pro
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <span className={`w-2 h-2 rounded-full ${wsStatus === 'Connected' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></span>
              Live: {wsStatus}
            </div>
            <input 
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="bg-[#080B0F] border border-[#C4A050]/30 text-white rounded-lg px-4 py-1.5 focus:outline-none focus:border-[#C4A050] transition-colors w-32 font-mono"
              placeholder="TICKER"
            />
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Hero Section */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-[#0C0F14] to-[#111520] border border-[#C4A050]/20 p-8 mb-8">
          <div className="absolute top-0 right-0 w-64 h-64 bg-radial-gradient from-[#C4A050]/10 to-transparent -translate-y-1/2 translate-x-1/2 rounded-full blur-2xl"></div>
          
          <div className="flex justify-between items-start relative z-10">
            <div>
              <h1 className="text-4xl font-serif text-[#F0D888] mb-2">{ticker}</h1>
              <p className="text-sm text-gray-400 mb-4">Real-time Advanced ML Analysis</p>
              <div className="flex gap-2">
                <span className="px-3 py-1 rounded-full text-xs font-semibold bg-[#C4A050]/10 border border-[#C4A050]/30 text-[#C4A050]">LSTM + XGBoost</span>
                <span className="px-3 py-1 rounded-full text-xs font-semibold bg-[#C4A050]/10 border border-[#C4A050]/30 text-[#C4A050]">Alpaca Executions</span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-4xl font-mono text-[#F0D888]">${currentPrice ? currentPrice.toFixed(2) : "---"}</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Main Chart */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-[#0E1118] border border-[#C4A050]/10 rounded-2xl p-6 hover:border-[#C4A050]/30 transition-colors">
              <h2 className="text-lg font-serif text-[#C4A050] mb-6 flex items-center gap-2">
                <TrendingUp className="w-5 h-5" /> Live Tick Chart
              </h2>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={priceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="time" stroke="#555" fontSize={11} tickMargin={10} />
                    <YAxis domain={['auto', 'auto']} stroke="#555" fontSize={11} width={60} tickFormatter={(v) => `$${v.toFixed(2)}`} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0C0F14', borderColor: 'rgba(196,160,80,0.3)', color: '#E8E0D0' }}
                      itemStyle={{ color: '#F0D888' }}
                    />
                    <Line type="monotone" dataKey="price" stroke="#4ADE80" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Trading Panel */}
            <div className="bg-[#0E1118] border border-[#C4A050]/10 rounded-2xl p-6">
               <h2 className="text-lg font-serif text-[#C4A050] mb-4 flex items-center gap-2">
                <DollarSign className="w-5 h-5" /> Automated Execution
              </h2>
              <p className="text-sm text-gray-400 mb-6">Signals generated by the ML ensemble can automatically trigger real trades via the Alpaca API.</p>
              <div className="flex gap-4">
                <button className="flex-1 bg-[#4ADE80]/10 border border-[#4ADE80]/30 text-[#4ADE80] font-semibold py-3 rounded-xl hover:bg-[#4ADE80]/20 transition-all flex items-center justify-center gap-2">
                  <ArrowUpRight className="w-4 h-4" /> Go Long (Market)
                </button>
                <button className="flex-1 bg-[#F87171]/10 border border-[#F87171]/30 text-[#F87171] font-semibold py-3 rounded-xl hover:bg-[#F87171]/20 transition-all flex items-center justify-center gap-2">
                  <ArrowDownRight className="w-4 h-4" /> Go Short (Market)
                </button>
              </div>
            </div>
          </div>

          {/* Sidebar / Prediction */}
          <div className="space-y-6">
            <div className="bg-[#0E1118] border border-[#C4A050]/10 rounded-2xl p-6">
              <h2 className="text-lg font-serif text-[#C4A050] mb-4 flex items-center gap-2">
                <Crosshair className="w-5 h-5" /> AI Prediction
              </h2>
              <button 
                onClick={generatePrediction}
                disabled={loading}
                className="w-full bg-[#C4A050]/10 border border-[#C4A050]/40 text-[#C4A050] font-semibold py-3 rounded-xl hover:bg-[#C4A050]/20 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {loading ? "Training Ensemble..." : "Generate 7-Day Forecast"} <ArrowRight className="w-4 h-4" />
              </button>

              {prediction && (
                <div className="mt-8 space-y-4">
                  <div className="p-4 rounded-xl border border-white/5 bg-white/5">
                    <div className="text-xs text-gray-500 uppercase tracking-wider font-semibold mb-1">Target Price</div>
                    <div className="text-3xl font-mono text-[#F0D888]">${prediction.predicted_price.toFixed(2)}</div>
                    <div className={`text-sm mt-1 font-mono ${prediction.predicted_return >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {prediction.predicted_return >= 0 ? '+' : ''}{prediction.predicted_return.toFixed(2)}%
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 rounded-xl border border-white/5 bg-white/5">
                      <div className="text-xs text-gray-500 uppercase tracking-wider font-semibold mb-1">Signal</div>
                      <div className={`text-lg font-bold tracking-widest ${
                        prediction.signal === 'BUY' ? 'text-green-400' : 
                        prediction.signal === 'SELL' ? 'text-red-400' : 'text-[#C4A050]'
                      }`}>
                        {prediction.signal}
                      </div>
                    </div>
                    <div className="p-4 rounded-xl border border-white/5 bg-white/5">
                      <div className="text-xs text-gray-500 uppercase tracking-wider font-semibold mb-1">Confidence</div>
                      <div className="text-lg font-mono text-white">{(prediction.confidence * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
