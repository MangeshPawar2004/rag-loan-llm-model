import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

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
        """Create the custom prompt template for loan recommendations."""
        template = """
        You are a financial loan recommendation expert with extensive knowledge of various loan products.
        
        INSTRUCTIONS:
        - Use ONLY the information provided in the context below
        - Recommend the most suitable loan type based on the user's specific needs
        - If multiple loans are suitable, rank them by relevance
        - Be specific about eligibility requirements and documentation
        - If information is insufficient, clearly state what additional details are needed
        
        Context (Available Loan Products): 
        {context}
        
        User Query: {question}
        
        RESPONSE FORMAT:
        1. **Recommended Loan**: [Primary recommendation with brief reasoning]
        2. **Key Features**: [Interest rate, tenure, amount range]
        3. **Eligibility**: [Age, income, credit score requirements]
        4. **Required Documents**: [List essential documents]
        5. **Additional Notes**: [Any important considerations or alternatives]
        
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
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query and return recommendations with sources."""
        if not user_query or not user_query.strip():
            return {
                "result": "Please provide a valid loan-related question.",
                "source_documents": [],
                "success": False
            }
        
        try:
            logger.info(f"Processing query: {user_query[:50]}...")
            response = self.qa_chain.invoke({"query": user_query})
            
            # Enhance response with metadata
            enhanced_response = {
                "result": response["result"],
                "source_documents": response["source_documents"],
                "success": True,
                "query": user_query,
                "num_sources": len(response["source_documents"])
            }
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "result": f"Sorry, I encountered an error while processing your query: {str(e)}",
                "source_documents": [],
                "success": False
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