import os
import asyncio
from typing import List, Dict, Any
from loguru import logger
import anthropic
from dotenv import load_dotenv
load_dotenv()

class LLMSwarm:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key) if self.api_key else None

    async def _simulate_agent(self, role: str, ticker: str, context: str) -> str:
        """Fallback simulation if no API key is present."""
        await asyncio.sleep(1.5)
        if role == "Quant":
            return f"Technicals for {ticker} show strong momentum. MACD crossover detected. I vote BUY."
        elif role == "Fundamental":
            return f"Earnings report for {ticker} indicates 15% YoY revenue growth, but P/E is slightly high. I'm cautiously optimistic. Vote HOLD."
        elif role == "Sentiment":
            return f"Social volume for {ticker} is incredibly high today. Retail traders are bullish. I vote BUY."
        else:
            return f"Considering the mixed fundamental and quant signals, position sizing should be kept to 5% of portfolio. Proceed with BUY."

    async def run_committee_debate(self, ticker: str) -> List[Dict[str, str]]:
        """
        Orchestrates a debate between 4 specialized AI agents.
        Returns a transcript of the debate.
        """
        logger.info(f"Starting LLM Swarm debate for {ticker}")
        
        roles = [
            {"name": "Quant", "prompt": "You are a quantitative trading AI. Analyze technical indicators and price action."},
            {"name": "Fundamental", "prompt": "You are a fundamental analysis AI. Evaluate earnings, P/E ratios, and macroeconomics."},
            {"name": "Sentiment", "prompt": "You are a sentiment analysis AI. Gauge social media trends and news sentiment."},
            {"name": "Risk Manager", "prompt": "You are a strict risk management AI. Review the previous opinions and give a final verdict."}
        ]

        transcript = []
        debate_context = f"The stock in question is {ticker}. Please provide your analysis."

        for role in roles:
            if self.client:
                try:
                    # In a real scenario, we'd pull actual data into the prompt
                    full_prompt = f"{role['prompt']} {debate_context}"
                    response = await self.client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=150,
                        system="You are an expert financial AI agent. Be concise and professional.",
                        messages=[{"role": "user", "content": full_prompt}]
                    )
                    content = response.content[0].text
                except Exception as e:
                    logger.error(f"Anthropic API error for {role['name']}: {e}")
                    content = await self._simulate_agent(role["name"], ticker, debate_context)
            else:
                content = await self._simulate_agent(role["name"], ticker, debate_context)

            transcript.append({
                "agent": role["name"],
                "message": content
            })
            
            # Append this agent's opinion to the context so the next agent can see it
            debate_context += f"\n\n{role['name']} said: {content}"

        return transcript

    async def parse_voice_intent(self, transcript: str) -> Dict[str, Any]:
        """
        Parses voice transcript into structured JSON intent.
        Example: "buy 5 shares of apple" -> {"action": "buy", "ticker": "AAPL", "qty": 5}
        """
        logger.info(f"Parsing voice intent for: '{transcript}'")
        
        prompt = f"""
        You are an AI financial copilot. Extract the trading intent from the user's voice command.
        Respond ONLY with a valid JSON object. Do not include markdown formatting or extra text.
        
        The JSON must match this schema:
        {{
            "action": "buy" | "sell" | "switch_market" | "view_ticker" | "unknown",
            "ticker": "TICKER SYMBOL (e.g. AAPL, BTC/USD)",
            "qty": number (or null if not applicable),
            "market": "TRADFI" | "CRYPTO" (only if action is switch_market, else null)
        }}
        
        User command: "{transcript}"
        """
        
        if self.client:
            try:
                response = await self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=100,
                    system="You are a JSON-only API.",
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text
                import json
                return json.loads(text)
            except Exception as e:
                logger.error(f"Anthropic API error for voice intent: {e}")
                pass
                
        # Fallback simulation if no API key or API fails
        text = transcript.lower()
        if "buy" in text:
            qty = 1
            words = text.split()
            for w in words:
                if w.isdigit():
                    qty = int(w)
                    break
            ticker = "AAPL" if "apple" in text else ("BTC/USD" if "bitcoin" in text else "TSLA")
            return {"action": "buy", "ticker": ticker, "qty": qty, "market": None}
        elif "crypto" in text or "bitcoin" in text:
            return {"action": "switch_market", "ticker": "BTC/USD", "qty": None, "market": "CRYPTO"}
        elif "apple" in text or "stock" in text:
            return {"action": "switch_market", "ticker": "AAPL", "qty": None, "market": "TRADFI"}
        
        return {"action": "unknown", "ticker": None, "qty": None, "market": None}

agent_swarm = LLMSwarm()

