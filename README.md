# 🌾 Farmer's Schemes Chatbot

A sophisticated AI-powered chatbot built using RAG (Retrieval Augmented Generation) technology to help Indian farmers access information about government schemes and agricultural programs.

## 🎯 Purpose

This chatbot serves as a bridge between farmers and complex government scheme information, providing:

- Instant access to agricultural scheme details
- Easy-to-understand explanations of eligibility criteria
- Application process guidance
- Information about financial assistance programs
- Details about subsidies and benefits

## 🛠️ Technical Architecture

### Core Components

- **Frontend**: Streamlit-based interactive web interface
- **Backend**: LangChain framework with Google's Gemini Pro LLM
- **RAG Implementation**:
  - FAISS vector store for efficient similarity search
  - Google's Embedding model for text embeddings
  - Custom prompt templates for contextual responses

### Data Processing

- PDF documents are processed and split into optimal chunks
- Embeddings are created and stored in FAISS vector store
- Real-time retrieval and response generation

## 🚀 Features

- **Natural Language Understanding**: Understands queries in simple, conversational language
- **Context-Aware Responses**: Provides accurate information based on official documentation
- **User-Friendly Interface**: Clean, intuitive design with modern styling
- **Real-Time Processing**: Quick response generation with relevant information
- **Source-Based Answers**: All responses are grounded in official government documents

## 💻 Technical Requirements

```python
Python >= 3.11
Dependencies:
- langchain >= 0.3.24
- langchain-google-genai >= 2.1.3
- langchain-community >= 0.3.22
- faiss-cpu >= 1.10.0
- streamlit >= 1.44.1
- python-dotenv >= 1.1.0
```

## 🛠️ Setup and Installation

1. Clone the repository

```bash
git clone [repository-url]
cd farmers-schemes-chatbot
```

2. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Set up environment variables

```bash
# Create .env file and add:
GOOGLE_API_KEY=your_api_key_here
```

5. Run the application

```bash
streamlit run app.py
```

## 🎯 Usage Examples

The chatbot can answer questions like:

- "What are the eligibility criteria for PM Kisan Samman Nidhi?"
- "How can I apply for Kisan Credit Card?"
- "Tell me about crop insurance schemes"
- "What documents are needed for agricultural loans?"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Mangesh Pawar - Initial work - https://github.com/MangeshPawar2004

## 🙏 Acknowledgments

- Google Gemini  for LLM capabilities
- Streamlit for the web interface framework
- LangChain for the RAG implementation
- FAISS for vector storage and similarity search

## 📞 Support

For support, email mangeshpawarmrp2004@gmail.com or open an issue in the repository.

---

Developed with ❤️ for Indian Farmers
