"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, MicOff, Loader2 } from "lucide-react";

interface VoiceCopilotProps {
  onIntentParsed: (intent: any) => void;
}

export default function VoiceCopilot({ onIntentParsed }: VoiceCopilotProps) {
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcript, setTranscript] = useState("");
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    // Initialize Web Speech API
    if (typeof window !== "undefined" && ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = "en-US";

      recognition.onstart = () => {
        setIsListening(true);
        setTranscript("");
      };

      recognition.onresult = (event: any) => {
        const current = event.resultIndex;
        const result = event.results[current];
        const transcriptText = result[0].transcript;
        setTranscript(transcriptText);
      };

      recognition.onerror = (event: any) => {
        console.error("Speech recognition error", event.error);
        setIsListening(false);
      };

      recognition.onend = async () => {
        setIsListening(false);
        // Process the final transcript if we have one
        const finalTranscript = transcriptRef.current;
        if (finalTranscript.trim().length > 0) {
          await processVoiceCommand(finalTranscript);
        }
      };

      recognitionRef.current = recognition;
    }
  }, []);

  // Use ref for transcript inside onend callback to get the latest state
  const transcriptRef = useRef(transcript);
  useEffect(() => {
    transcriptRef.current = transcript;
  }, [transcript]);

  const processVoiceCommand = async (text: string) => {
    setIsProcessing(true);
    try {
      const res = await fetch("http://localhost:8000/voice/intent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: text })
      });
      if (res.ok) {
        const intent = await res.json();
        onIntentParsed(intent);
      }
    } catch (err) {
      console.error("Error processing voice command", err);
    }
    setIsProcessing(false);
    setTimeout(() => setTranscript(""), 3000);
  };

  const toggleListen = () => {
    if (!recognitionRef.current) {
      alert("Voice recognition is not supported in this browser.");
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
    } else {
      recognitionRef.current.start();
    }
  };

  return (
    <div className="fixed bottom-8 right-8 z-50 flex flex-col items-end gap-4 pointer-events-none">
      
      {/* Transcript Bubble */}
      <AnimatePresence>
        {(isListening || isProcessing || transcript) && (
          <motion.div 
            initial={{ opacity: 0, y: 10, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="bg-[#0E1118]/90 backdrop-blur-md border border-[#C4A050]/40 rounded-2xl p-4 shadow-2xl max-w-xs pointer-events-auto"
          >
            <div className="text-xs text-[#C4A050] uppercase tracking-widest font-semibold mb-1 flex items-center gap-2">
              {isProcessing ? <><Loader2 className="w-3 h-3 animate-spin" /> Processing AI...</> : "Voice Co-Pilot"}
            </div>
            <div className="text-sm text-gray-200">
              {transcript || "Listening..."}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Floating Action Button */}
      <motion.button
        onClick={toggleListen}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        className={`pointer-events-auto w-16 h-16 rounded-full flex items-center justify-center shadow-[0_0_30px_rgba(196,160,80,0.3)] transition-all relative ${
          isListening 
            ? "bg-red-500/20 border-2 border-red-500 text-red-500 shadow-[0_0_30px_rgba(239,68,68,0.5)]" 
            : "bg-[#C4A050]/10 border border-[#C4A050]/50 text-[#C4A050] hover:bg-[#C4A050]/20"
        }`}
      >
        {isListening && (
          <span className="absolute inset-0 rounded-full animate-ping bg-red-500/30"></span>
        )}
        {isListening ? <Mic className="w-6 h-6" /> : <MicOff className="w-6 h-6" />}
      </motion.button>
    </div>
  );
}
