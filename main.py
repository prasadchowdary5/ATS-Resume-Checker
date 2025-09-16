import streamlit as st
import pandas as pd
import numpy as np
import re
from docx import Document
import PyPDF2
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Manual stop words list (no NLTK required)
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', 
    "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

# Page configuration
st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .high-score {
        color: #28a745;
        font-weight: bold;
    }
    .medium-score {
        color: #ffc107;
        font-weight: bold;
    }
    .low-score {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Job role descriptions (keywords and skills)
JOB_ROLES = {
    "Software Engineer": {
        "keywords": ["python", "java", "javascript", "c++", "sql", "git", "docker", "kubernetes", 
                    "aws", "azure", "react", "node", "django", "flask", "api", "microservices"],
        "description": "Develops software applications and systems using various programming languages and technologies."
    },
    "Data Scientist": {
        "keywords": ["python", "r", "sql", "machine learning", "statistics", "pandas", "numpy", 
                    "tensorflow", "pytorch", "data visualization", "big data", "hadoop", "spark"],
        "description": "Analyzes complex data sets to extract insights and build predictive models."
    },
    "Product Manager": {
        "keywords": ["product strategy", "roadmap", "agile", "scrum", "user stories", "market research", 
                    "stakeholder management", "product launch", "metrics", "customer discovery"],
        "description": "Manages product development from conception to launch, working with cross-functional teams."
    },
    "UX Designer": {
        "keywords": ["user research", "wireframing", "prototyping", "figma", "sketch", "adobe xd", 
                    "usability testing", "user flows", "design thinking", "ui design"],
        "description": "Designs user interfaces and experiences for digital products."
    },
    "Marketing Manager": {
        "keywords": ["digital marketing", "seo", "sem", "social media", "content strategy", "campaign management", 
                    "analytics", "brand management", "market research", "email marketing"],
        "description": "Develops and implements marketing strategies to promote products or services."
    }
}

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file (PDF or DOCX)"""
    text = ""
    
    if uploaded_file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
            
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            doc = Document(io.BytesIO(uploaded_file.read()))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return None
    else:
        st.error("Unsupported file format. Please upload PDF or DOCX.")
        return None
        
    return text.lower()

def preprocess_text(text):
    """Preprocess text by removing special characters and stop words"""
    if not text:
        return ""
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Simple tokenization and stop word removal using our manual list
    words = text.lower().split()
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    
    return ' '.join(filtered_words)

def calculate_ats_score(resume_text, job_role):
    """Calculate ATS score based on keyword matching and similarity"""
    if not resume_text:
        return {
            "final_score": 0,
            "keyword_score": 0,
            "similarity_score": 0,
            "found_keywords": [],
            "missing_keywords": JOB_ROLES[job_role]["keywords"]
        }
    
    # Preprocess resume text
    processed_resume = preprocess_text(resume_text)
    
    # Get job role keywords
    job_keywords = JOB_ROLES[job_role]["keywords"]
    job_description = " ".join(job_keywords)
    
    # Calculate keyword match score
    found_keywords = []
    for keyword in job_keywords:
        if keyword in processed_resume:
            found_keywords.append(keyword)
    
    keyword_score = (len(found_keywords) / len(job_keywords)) * 100 if job_keywords else 0
    
    # Calculate TF-IDF similarity
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([processed_resume, job_description])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    except Exception as e:
        st.warning(f"Similarity calculation limited: {e}")
        similarity_score = keyword_score * 0.8  # Fallback score
    
    # Combined score (weighted average)
    final_score = (keyword_score * 0.6) + (similarity_score * 0.4)
    
    return {
        "final_score": round(final_score, 1),
        "keyword_score": round(keyword_score, 1),
        "similarity_score": round(similarity_score, 1),
        "found_keywords": found_keywords,
        "missing_keywords": list(set(job_keywords) - set(found_keywords))
    }

def get_score_color(score):
    """Return color based on score"""
    if score >= 70:
        return "high-score"
    elif score >= 40:
        return "medium-score"
    else:
        return "low-score"

def main():
    st.markdown('<h1 class="main-header">📄 Resume ATS Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar for job role selection
    with st.sidebar:
        st.header("Job Role Selection")
        selected_role = st.selectbox(
            "Choose a job role:",
            list(JOB_ROLES.keys()),
            help="Select the job role you're applying for"
        )
        
        st.header("About")
        st.info("""
        This tool analyzes your resume against specific job roles and provides an ATS (Applicant Tracking System) compatibility score.
        
        **How it works:**
        - Upload your resume (PDF or DOCX)
        - Select a target job role
        - Get instant feedback on keyword matching and overall compatibility
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Your Resume")
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=["pdf", "docx"],
            help="Supported formats: PDF, DOCX"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Display job role information
            st.subheader("🎯 Selected Job Role")
            st.write(f"**{selected_role}**")
            st.write(JOB_ROLES[selected_role]["description"])
            
            # Store results in session state to persist across reruns
            if 'results' not in st.session_state:
                st.session_state.results = None
            
            # Analyze button
            if st.button("🚀 Analyze Resume", type="primary"):
                with st.spinner("Analyzing your resume..."):
                    # Extract text from resume
                    resume_text = extract_text_from_file(uploaded_file)
                    
                    if resume_text:
                        # Calculate ATS score
                        results = calculate_ats_score(resume_text, selected_role)
                        st.session_state.results = results
                        
                        # Display results
                        st.subheader("📊 Analysis Results")
                        
                        # Score card
                        score_color = get_score_color(results["final_score"])
                        st.markdown(f"""
                        <div class="score-card">
                            <h3 class="{score_color}">Overall ATS Score: {results["final_score"]}%</h3>
                            <p>Keyword Match: {results["keyword_score"]}%</p>
                            <p>Content Similarity: {results["similarity_score"]}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Interpretation
                        if results["final_score"] >= 70:
                            st.success("🎉 Excellent! Your resume is well-optimized for this role.")
                        elif results["final_score"] >= 40:
                            st.warning("⚠️ Good start, but consider adding more relevant keywords.")
                        else:
                            st.error("❌ Your resume needs significant improvement for this role.")
    
    with col2:
        if uploaded_file is not None:
            # Check if results exist in session state
            if st.session_state.results is not None:
                results = st.session_state.results
                
                # Display keyword analysis
                st.subheader("🔍 Keyword Analysis")
                
                # Found keywords
                if results["found_keywords"]:
                    st.success("✅ **Keywords Found:**")
                    st.write(", ".join(results["found_keywords"]))
                else:
                    st.info("No matching keywords found.")
                
                # Missing keywords
                if results["missing_keywords"]:
                    st.error("❌ **Keywords to Add:**")
                    st.write(", ".join(results["missing_keywords"]))
                
                # Tips for improvement
                st.subheader("💡 Improvement Tips")
                tips = [
                    "Include specific technologies mentioned in the job description",
                    "Use industry-standard terminology",
                    "Quantify your achievements with numbers",
                    "Tailor your resume for each specific job application",
                    "Include both hard and soft skills relevant to the role"
                ]
                
                for i, tip in enumerate(tips, 1):
                    st.write(f"{i}. {tip}")
            
            else:
                st.info("Click 'Analyze Resume' to see your ATS score and recommendations.")
        
        else:
            # Default content
            st.subheader("ℹ️ How to Use")
            st.write("""
            1. **Upload** your resume (PDF or DOCX format)
            2. **Select** the target job role from the sidebar
            3. **Click** 'Analyze Resume' to get your ATS score
            4. **Review** the analysis and improvement suggestions
            """)
            
            st.subheader("📈 What is ATS?")
            st.write("""
            Applicant Tracking Systems (ATS) are software used by employers to:
            - Screen resumes before human review
            - Parse and categorize applicant information
            - Rank candidates based on keyword matching
            - Filter applications based on specific criteria
            
            Optimizing your resume for ATS can significantly increase your chances of getting an interview.
            """)

if __name__ == "__main__":
    main()
