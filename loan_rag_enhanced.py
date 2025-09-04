import os
import logging
import re
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanRAGSystem:
    """A RAG system for loan recommendations using Google Gemini and FAISS."""
    
    def __init__(self, vectorstore_path: str = "vectorstore/db_faiss"):
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.vectorstore_path = vectorstore_path
        
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self._initialize_models()
        self._load_vectorstore()
        self._setup_chain()
    
    def _initialize_models(self):
        """Initialize embedding and language models."""
        try:
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                google_api_key=self.google_api_key, 
                temperature=0.2
            )
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _load_vectorstore(self):
        """Load the FAISS vectorstore."""
        try:
            self.db = FAISS.load_local(
                self.vectorstore_path, 
                self.embedding_model, 
                allow_dangerous_deserialization=True
            )
            logger.info("Vectorstore loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vectorstore: {e}")
            raise
    
    def _get_custom_prompt(self) -> PromptTemplate:
        """Create the custom prompt template for loan recommendations with structured output."""
        template = """
        You are a financial loan recommendation expert with extensive knowledge of various loan products.
        
        INSTRUCTIONS:
        - Use ONLY the information provided in the context below.
        - Recommend the most suitable loan type(s) based on the user's specific needs.
        - Present the information in clear, concise, and structured bullet points or sub-sections.
        - If multiple loans are suitable, present them clearly, indicating their differences and benefits.
        - Be specific about eligibility requirements and documentation.
        - If information is insufficient, clearly state what additional details are needed.
        - If you can extract comparative numerical data (e.g., interest rates, loan amounts for different banks/products), please present it clearly, perhaps in a simple list or table-like format within the text, so it can potentially be parsed for a graph. Indicate the bank/product name for each data point.

        Context (Available Loan Products): 
        {context}
        
        User Query: {question}
        
        RESPONSE FORMAT:
        Please structure your response using the following markdown format:
        
        **🎯 Recommended Loan Options:**
        *   **Loan Type 1 Name:** [Brief overview of this loan type]
            *   **Key Features:**
                *   Interest Rate Range: [e.g., 8.0% - 10.5% (Bank A), 8.2% - 10.7% (Bank B)]
                *   Maximum Loan Amount: [e.g., Up to ₹50 lakhs (Bank A), Up to ₹75 lakhs (Bank B)]
                *   Tenure Options: [e.g., Up to 30 years]
            *   **Eligibility Criteria:**
                *   Age: [e.g., 21-65 years]
                *   Monthly Income: [e.g., ₹25,000 minimum]
                *   Credit Score: [e.g., 700+ preferred]
            *   **Required Documents:**
                *   [List essential documents]
            *   **Why it's suitable:** [Brief explanation]

        *   **Loan Type 2 Name (if applicable):** [Brief overview]
            *   **Key Features:** ...
            *   **Eligibility Criteria:** ...
            *   **Required Documents:** ...
            *   **Why it's suitable:** ...

        **📝 General Eligibility & Documentation (if not specific to a loan type):**
        *   [General requirements]

        **💡 Additional Considerations:**
        *   [Tips, next steps, or missing information if any]

        If the context doesn't contain sufficient information to answer the query, respond with:
        "I don't have enough information in my knowledge base to provide a specific recommendation for this query. Please provide more details about [specific missing information]."
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _setup_chain(self):
        """Setup the RetrievalQA chain."""
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.db.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 5, 'fetch_k': 10}
                ),
                return_source_documents=True,
                chain_type_kwargs={'prompt': self._get_custom_prompt()},
                input_key="query",
                output_key="result"
            )
            logger.info("QA chain setup successfully")
        except Exception as e:
            logger.error(f"Failed to setup QA chain: {e}")
            raise

    def _extract_graph_data(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Extracts numerical data for graphing from the LLM's response.
        This is a heuristic approach and might need refinement based on actual LLM output.
        It looks for patterns like 'Bank X: Y - Z%' or 'Bank P: Up to ₹Amount'.
        """
        graph_data = []
        
        # Regex to find interest rates for different banks/products
        # Example: "8.0% - 10.5% (Bank A)" or "Bank B: 8.2% - 10.7%"
        # Looking for "Bank Name: Interest Rate Range" or "Interest Rate Range (Bank Name)"
        interest_rate_pattern = re.compile(
            r"(?:(\w[\w\s&.]*?):\s*)?([\d.]+\%)(?:\s*-\s*([\d.]+\%))?(?:\s*\(([\w\s&.]*?)\))?", 
            re.IGNORECASE
        )
        
        # Regex to find loan amounts for different banks/products
        # Example: "Up to ₹50 lakhs (Bank A)" or "Bank C: Max ₹75 lakhs"
        loan_amount_pattern = re.compile(
            r"(?:(\w[\w\s&.]*?):\s*(?:Up to|Max)\s*)?₹([\d,]+)\s*(?:lakhs|crores)?(?:\s*\(([\w\s&.]*?)\))?", 
            re.IGNORECASE
        )

        lines = response_text.split('\n')
        current_loan_type = "Overall" # Default for general recommendations

        for line in lines:
            if "**Loan Type" in line:
                match = re.search(r"\*\*Loan Type \d+ Name:\*\* (.+)", line)
                if match:
                    current_loan_type = match.group(1).strip()
                continue
            
            # Extract interest rates
            for match in interest_rate_pattern.finditer(line):
                bank_name = match.group(1) or match.group(4)
                if not bank_name: # Try to infer from current loan type if no specific bank
                    bank_name = current_loan_type.replace("Loan Type", "Product") 
                
                if bank_name:
                    try:
                        rate_min = float(match.group(2).replace('%', '').strip())
                        rate_max = float(match.group(3).replace('%', '').strip()) if match.group(3) else rate_min
                        graph_data.append({
                            'Bank/Product': bank_name,
                            'Metric': 'Interest Rate (Min %)',
                            'Value': rate_min
                        })
                        if rate_max != rate_min:
                             graph_data.append({
                                'Bank/Product': bank_name,
                                'Metric': 'Interest Rate (Max %)',
                                'Value': rate_max
                            })
                    except (ValueError, TypeError):
                        pass # Ignore if conversion fails

            # Extract loan amounts
            for match in loan_amount_pattern.finditer(line):
                bank_name = match.group(1) or match.group(3)
                if not bank_name:
                    bank_name = current_loan_type.replace("Loan Type", "Product")
                
                if bank_name:
                    try:
                        amount_str = match.group(2).replace(',', '')
                        amount = float(amount_str)
                        if "lakhs" in line.lower():
                            amount *= 100000
                        elif "crores" in line.lower():
                            amount *= 10000000
                        graph_data.append({
                            'Bank/Product': bank_name,
                            'Metric': 'Max Loan Amount (₹)',
                            'Value': amount
                        })
                    except (ValueError, TypeError):
                        pass # Ignore if conversion fails
        
        # Filter out generic entries if more specific ones exist
        if any(d['Bank/Product'] != "Overall" for d in graph_data):
            graph_data = [d for d in graph_data if d['Bank/Product'] != "Overall"]
            
        return graph_data


    def query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query and return recommendations with sources and graph data."""
        if not user_query or not user_query.strip():
            return {
                "result": "Please provide a valid loan-related question.",
                "source_documents": [],
                "success": False,
                "graph_data": []
            }
        
        try:
            logger.info(f"Processing query: {user_query[:50]}...")
            response = self.qa_chain.invoke({"query": user_query})
            
            # Extract graph data from the LLM's structured response
            graph_data = self._extract_graph_data(response["result"])

            # Enhance response with metadata
            enhanced_response = {
                "result": response["result"],
                "source_documents": response["source_documents"],
                "success": True,
                "query": user_query,
                "num_sources": len(response["source_documents"]),
                "graph_data": graph_data
            }
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "result": f"Sorry, I encountered an error while processing your query: {str(e)}",
                "source_documents": [],
                "success": False,
                "graph_data": []
            }
    
    def get_similar_documents(self, query: str, k: int = 3) -> list:
        """Get similar documents without running through the full chain."""
        try:
            return self.db.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        print("🏦 Loan Recommendation System")
        print("=" * 50)
        print("Ask me about loans, eligibility, interest rates, or documents required.")
        print("Type 'quit', 'exit', or 'bye' to end the session.\n")
        
        while True:
            try:
                user_query = input("\n💬 Your question: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'bye', '']:
                    print("\n👋 Thank you for using the Loan Recommendation System!")
                    break
                
                print("\n🔍 Searching for relevant information...")
                response = self.query(user_query)
                
                if response["success"]:
                    print(f"\n✅ **Answer:**\n{response['result']}")
                    
                    if response["graph_data"]:
                        print("\n📊 **Extracted Data for Graphing:**")
                        for item in response["graph_data"]:
                            print(f"  - {item['Bank/Product']}: {item['Metric']} = {item['Value']}")

                    if response["source_documents"]:
                        print(f"\n📚 **Sources Used ({response['num_sources']} documents):**")
                        for i, doc in enumerate(response["source_documents"], 1):
                            title = doc.metadata.get('source', doc.metadata.get('title', 'Unknown Source'))
                            content_preview = doc.page_content[:100].replace('\n', ' ')
                            print(f"{i}. {title}")
                            print(f"   Preview: {content_preview}...")
                            print()
                else:
                    print(f"\n❌ **Error:** {response['result']}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error in interactive chat: {e}")
                print(f"\n❌ An unexpected error occurred: {e}")


def main():
    """Main function to run the loan RAG system."""
    try:
        print("🚀 Initializing Loan RAG System...")
        loan_system = LoanRAGSystem()
        
        loan_system.interactive_chat()
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        print(f"❌ System initialization failed: {e}")
        print("\nPlease check:")
        print("1. GOOGLE_API_KEY is set in your .env file")
        print("2. vectorstore/db_faiss directory exists and contains valid data")
        print("3. All required packages are installed")


if __name__ == "__main__":
    main()