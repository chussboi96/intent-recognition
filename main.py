from typing import Annotated, TypedDict, List, Dict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.graph.message import add_messages
import operator
import uuid
import torch
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import OllamaLLM
from base import BaseChatbot
from stt_hf import VoiceProcessor 
from query_classifier import QueryClassifier

class ChatPromptTemplates:
    SYSTEM_PROMPTS = {
        'classify': """You are a classification expert assistant. Your task is to:
- Analyze the input thoroughly
- Consider multiple perspectives
- Use clear, structured classification
- Provide specific examples when relevant
- Be concise but comprehensive

Input to analyze: {input}""",
            
        'factcheck': """You are a fact-checking expert. For this query:
- Verify the information thoroughly
- Check against reliable sources
- Provide evidence-based analysis
- Be objective and accurate
- Explain your reasoning clearly

Content to verify: {input}""",
            
        'summarize': """You are a summarization expert. Your task is to:
- Identify and extract key points
- Maintain essential context and details
- Create a clear, structured summary
- Focus on the most relevant information
- Present information concisely

Content to summarize: {input}""",
            
        'analyze': """You are an analytical assistant. Your task is to:
- Break down the topic systematically
- Consider multiple viewpoints
- Provide logical analysis
- Support conclusions with reasoning
- Address potential counterarguments

Topic to analyze: {input}""",
            
        'detail': """You are a detailed explanation expert. For this topic:
- Provide comprehensive, accurate information
- Break down complex concepts
- Give clear, step-by-step explanations
- Use relevant examples
- Ensure thorough understanding

Topic to explain: {input}""",
            
        'fallback': """You are a helpful assistant. For this query:
- Understand the context and intention
- Provide relevant, helpful information
- Use clear, simple language
- Focus on being practical and useful
- Ask for clarification if needed

Query to address: {input}"""
    }
    
    @staticmethod
    def get_prompt(intent: str, input_text: str) -> str:
        """Get formatted prompt based on intent"""
        template = ChatPromptTemplates.SYSTEM_PROMPTS.get(
            intent, 
            ChatPromptTemplates.SYSTEM_PROMPTS['fallback']
        )
        return template.format(input=input_text)


class ChatState(MessagesState):
    intent: str | None
    confidence: float | None
    # operator.add for accumulating results
    results: Annotated[List[Dict], operator.add]
    parallel_tasks: Annotated[List[str], operator.add]

class LangGraphChatbot(BaseChatbot):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.llm = OllamaLLM(model="qwen2.5:3b")
        self.setup_intent_classifier()
        self.setup_workflow()
        print("LangGraph Chatbot initialized successfully")

    def setup_intent_classifier(self):
        try:
            self.labels = ["classify", "factcheck", "summarize", "analyze", "detail"]
            self.intent_classifier = QueryClassifier(self.labels)
            model_path = r"C:\Users\LOQ\intent_classifier"
            self.intent_classifier.load_model(model_path)
            print("SpaCy intent classifier loaded successfully")
        except Exception as e:
            print(f"Error initializing spaCy intent classifier: {e}")
            raise

    def setup_workflow(self):
        workflow = StateGraph(state_schema=ChatState)
        
        # Add nodes
        workflow.add_node("detect_intent", self.detect_intent_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("factcheck", self.factcheck_node)
        workflow.add_node("classify", self.classify_node)
        workflow.add_node("detail", self.detail_node)
        workflow.add_node("summarize", self.summarize_node)
        workflow.add_node("combine_results", self.combine_results_node)
        
        workflow.add_edge(START, "detect_intent")
        
        def route_actions(state: ChatState) -> Sequence[str]:
            intent = state.get("intent")
            confidence = state.get("confidence")
            message = state["messages"][-1].content.lower()
            
            actions = {
                "classify": ["classify", "detail"],
                "factcheck": ["factcheck", "analyze"],
                "summarize": ["summarize", "analyze"],
                "analyze": ["analyze", "detail"],
                "detail": ["detail", "analyze"],
                "fallback": ["analyze"]
            }
            
            parallel_actions = list(actions.get(intent, ["analyze"]))
            
            if confidence and confidence < 0.7:
                if "analyze" not in parallel_actions:
                    parallel_actions.append("analyze")
                if "detail" not in parallel_actions:
                    parallel_actions.append("detail")
                    
            if any(word in message for word in ["fact", "true", "verify", "check"]):
                if "factcheck" not in parallel_actions:
                    parallel_actions.append("factcheck")
                    
            if any(word in message for word in ["summary", "summarize", "brief"]):
                if "summarize" not in parallel_actions:
                    parallel_actions.append("summarize")
            
            print(f"Selected parallel actions: {parallel_actions}")
            return parallel_actions

        action_nodes = ["analyze", "factcheck", "classify", "detail", "summarize"]
        workflow.add_conditional_edges(
            "detect_intent",
            route_actions,
            action_nodes
        )
        
        for node in action_nodes:
            workflow.add_edge(node, "combine_results")
        
        workflow.add_edge("combine_results", END)
        
        self.memory = MemorySaver()
        self.app = workflow.compile(checkpointer=self.memory)
        
        self.thread_id = str(uuid.uuid4())
        self.config = {
            "configurable": {
                "thread_id": self.thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": str(uuid.uuid4())
            }
        }
        print("Workflow setup completed")

    def detect_intent_node(self, state: ChatState) -> Dict:
        last_message = state["messages"][-1].content
        try:
            predictions = self.intent_classifier.predict(last_message)
            if predictions:
                sorted_predictions = dict(sorted(
                    predictions.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
                
                intent = next(iter(sorted_predictions))
                confidence = sorted_predictions[intent]
                
                print(f"\nDetected Intent: {intent}")
                print(f"Confidence Score: {confidence:.2f}")
                
                print("\nAll Intent Scores:")
                for intent_name, score in sorted_predictions.items():
                    print(f"- {intent_name}: {score:.2f}")
                
                return {
                    "intent": intent, 
                    "confidence": confidence,
                    "results": [],
                    "parallel_tasks": []
                }
            else:
                print("No confident predictions - using fallback")
                return {
                    "intent": "fallback", 
                    "confidence": 0.0,
                    "results": [],
                    "parallel_tasks": []
                }
                
        except Exception as e:
            print(f"Error in intent detection: {e}")
            return {
                "intent": "fallback", 
                "confidence": 0.0,
                "results": [],
                "parallel_tasks": []
            }

    def analyze_node(self, state: ChatState) -> Dict:
        last_message = state["messages"][-1].content
        result = self.llm.invoke(ChatPromptTemplates.get_prompt("analyze", last_message))
        print("Analysis completed")
        return {
            "results": [{"type": "analysis", "content": str(result)}],
            "parallel_tasks": ["analyze"]
        }

    def factcheck_node(self, state: ChatState) -> Dict:
        last_message = state["messages"][-1].content
        result = self.llm.invoke(ChatPromptTemplates.get_prompt("factcheck", last_message))
        print("Fact-checking completed")
        return {
            "results": [{"type": "factcheck", "content": str(result)}],
            "parallel_tasks": ["factcheck"]
        }

    def classify_node(self, state: ChatState) -> Dict:
        last_message = state["messages"][-1].content
        result = self.llm.invoke(ChatPromptTemplates.get_prompt("classify", last_message))
        print("Classification completed")
        return {
            "results": [{"type": "classification", "content": str(result)}],
            "parallel_tasks": ["classify"]
        }

    def detail_node(self, state: ChatState) -> Dict:
        last_message = state["messages"][-1].content
        result = self.llm.invoke(ChatPromptTemplates.get_prompt("detail", last_message))
        print("Detailed analysis completed")
        return {
            "results": [{"type": "detail", "content": str(result)}],
            "parallel_tasks": ["detail"]
        }

    def summarize_node(self, state: ChatState) -> Dict:
        last_message = state["messages"][-1].content
        result = self.llm.invoke(ChatPromptTemplates.get_prompt("summarize", last_message))
        print("Summarization completed")
        return {
            "results": [{"type": "summary", "content": str(result)}],
            "parallel_tasks": ["summarize"]
        }

    def combine_results_node(self, state: ChatState) -> Dict:
        results = state.get("results", [])
        print(f"Combining results from {len(results)} parallel tasks")
        
        grouped_results = {}
        for result in results:
            result_type = result["type"]
            if result_type not in grouped_results:
                grouped_results[result_type] = []
            grouped_results[result_type].append(result["content"])
        
        combine_prompt = f"""Synthesize these parallel analysis results into a coherent response:

{chr(10).join(f'{k.title()}: {v[0]}' for k, v in grouped_results.items())}

Create a response that:
1. Integrates insights from all analyses
2. Maintains a clear and coherent structure
3. Emphasizes the most important findings
4. Provides a balanced perspective
"""
        
        final_response = self.llm.invoke(combine_prompt)
        print("Response generation completed")
        
        return {
            "messages": [AIMessage(content=str(final_response))],
            "parallel_tasks": ["combine"]
        }

    def run(self, input_text: str):
        try:
            input_message = HumanMessage(content=input_text)
            try:
                current_state = self.app.get_state(self.config)
                messages = current_state.get("messages", [])
                messages.append(input_message)
                initial_state = {
                    "messages": messages,
                    "intent": None,
                    "confidence": None,
                    "results": [],
                    "parallel_tasks": []
                }
            except Exception as e:
                print(f"Error getting current state: {e}")
                initial_state = {
                    "messages": [input_message],
                    "intent": None,
                    "confidence": None,
                    "results": [],
                    "parallel_tasks": []
                }
            
            print("Processing input...")
            for event in self.app.stream(initial_state, self.config, stream_mode="values"):
                if len(event["messages"]) > 0 and isinstance(event["messages"][-1], AIMessage):
                    yield event["messages"][-1].content
                    
        except Exception as e:
            print(f"Error in chatbot execution: {e}")
            yield "Sorry, I encountered an error processing your request."

    def generate_response(self, input_text: str) -> str:
        response_text = ""
        print("Generating response...")
        for response in self.run(input_text):
            response_text += response
        return self.post_process(response_text)

    def get_conversation_history(self) -> List[BaseMessage]:
        try:
            state = self.app.get_state(self.config)
            return state.get("messages", [])
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []

def main():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nInitializing system with device: {device}")        
        
        try:
            voice_processor = VoiceProcessor(
                model_id="openai/whisper-large-v3",
                device=device,
                logger=None
            )
        except Exception as e:
            print(f"Failed to initialize voice processor: {e}")
            return
            
        try:
            chatbot = LangGraphChatbot(logger=None)
        except Exception as e:
            print(f"Failed to initialize chatbot: {e}")
            return
            
        print("\nVoice Chat System Initialized.")
        print("The system will record for 10 seconds when you press Enter.\n")
        
        while True:
            try:
                input("\nPress Enter to start recording (or Ctrl+C to exit)...")
                
                print("\nRecording for 10 seconds...")
                voice_input = voice_processor.listen()
                
                if not voice_input:
                    print("\nNo speech detected, please try again...")
                    continue
                    
                print(f"\nYou said: {voice_input}")
                
                print("\nProcessing response...")
                response = chatbot.generate_response(voice_input)
                print(f"\nResponse: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGracefully shutting down...")
                break
            except Exception as e:
                print(f"Error during conversation: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                continue
                
    except Exception as e:
        print(f"Error initializing system: {str(e)}")
        return

if __name__ == "__main__":
    main()