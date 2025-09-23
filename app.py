import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import urlparse, parse_qs


# Hugging Face API Key

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_hf_key" 


# Streamlit App UI

st.title("Youtube Videos Insights")

video_url = st.text_input("Enter YouTube Video URL")
user_question = st.text_input("Ask a question about the video:")


def get_youtube_video_id(url: str) -> str:
    """
    Extract the video ID from a YouTube URL.
    Works for standard, short, and URLs with extra parameters.
    """
    parsed_url = urlparse(url)

    # Case 1: Standard URL (https://www.youtube.com/watch?v=VIDEO_ID)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]

    # Case 2: Short URL (https://youtu.be/VIDEO_ID)
    elif parsed_url.hostname in ["youtu.be"]:
        return parsed_url.path.lstrip("/")

    return None


if video_url and user_question:
    try:
        # 1. Fetch transcript
        
        video_id = get_youtube_video_id(video_url)
        
        ytt_api = YouTubeTranscriptApi() 
        fetched_transcript = ytt_api.fetch(video_id,languages=["en"])
        transcript = " ".join(chunk.text for chunk in fetched_transcript)


        # 2. Split into chunks
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])


        # 3. Embeddings + FAISS

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


        # 4. Hugging Face LLM

        llm_x = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-120b",
            task="text-generation",
            temperature=0.2,
            max_new_tokens=512
        )
        llm = ChatHuggingFace(llm=llm_x)

        # 5. Prompt

        prompt = PromptTemplate(
            template="""
              You are a helpful assistant.
              Answer ONLY from the provided transcript context.
              If the context is insufficient, just say you don't know.

              {context}
              Question: {question}
            """,
            input_variables=['context', 'question']
        )

        # 6. Retrieval + Answering

        retrieved_docs = retriever.invoke(user_question)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        # final_prompt = prompt.invoke({"context": context_text, "question": user_question})

        
        def format_docs(retrieved_docs):
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text
        
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        
        parser = StrOutputParser()
        
        main_chain = parallel_chain | prompt | llm | parser
        
        answer = main_chain.invoke(user_question)

        st.subheader("Answer")
        st.write(answer)
    
    
    except TranscriptsDisabled:
        st.error("No captions available for this video.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        

