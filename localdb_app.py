from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import re
import os
from difflib import SequenceMatcher
import anthropic
from openai import OpenAI
import asyncio
import json
import logging
import time
from fastapi.middleware.cors import CORSMiddleware
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ThemisCode API", description="API for generating and analyzing educational questions")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # CORS izin verilen kaynaklar
    allow_credentials=True,
    allow_methods=["*"],  # Tüm HTTP metodlarına izin ver
    allow_headers=["*"],  # Tüm başlıklara izin ver
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DIFFICULTY_LEVELS = {
    "BEGINNER": range(10, 51),      # %10-50: Basic/Easy questions
    "INTERMEDIATE": range(51, 81),   # %51-80: Medium/Moderate questions
    "ADVANCED": range(81, 101)       # %81-100: Hard/Academic questions
}
from typing import List
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter



    
    
from fastapi import UploadFile, File
from pdf_processor import PDFProcessor
from rag_manager import RAGManager

# Mevcut app tanımlamasından sonra
pdf_processor = PDFProcessor()
rag_manager = RAGManager()

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # PDF içeriğini oku
        pdf_content = await file.read()
        
        # PDF işleyici oluştur
        pdf_processor = PDFProcessor()
        
        # PDF'i asenkron olarak işle
        chunks = await asyncio.to_thread(pdf_processor.process_pdf, pdf_content)
        
        # RAG yöneticisine ekle
        rag_manager.add_documents(chunks)
        
        return {
            "success": True,
            "message": f"PDF başarıyla işlendi ve {len(chunks)} parça veritabanına eklendi",
            "chunk_count": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"PDF yükleme hatası: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
# Question Data Model
from pydantic import BaseModel
from typing import List, Optional, Dict

class QuestionData(BaseModel):
    question: str
    choices: str
    correct_answer: str
    explanation: str
    sources: Optional[List[str]] = None  # Kaynak bilgilerini ekledik

    class Config:
        from_attributes = True

class RegenerateQuestionRequest(BaseModel):
    main_topic: str
    sub_topics: List[str]
    content: str
    question_type: str
    difficulty: int
    model: str
    model_temperature: float = 0.3
    original_question: str
    openai_api_key: Optional[str] = "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"
    google_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = "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"
    similarity_threshold: float = 0.7

class RegenerateQuestionResponse(BaseModel):
    success: bool
    original_question: str
    question: str  # QuestionData yerine direkt string alanlar
    choices: str
    correct_answer: str
    explanation: str
    message: str
# Question Tracker class
class QuestionTracker:
    def __init__(self, similarity_threshold=0.7):
        self.questions = []
        self.question_hashes = set()
        self.similarity_threshold = similarity_threshold
    
    def hash_question(self, question):
        """Creates a unique hash value for the question"""
        normalized = re.sub(r'\s+', ' ', question.lower().strip())
        return hash(normalized)
    
    def is_similar(self, new_question):
        """Checks similarity of new question to existing ones"""
        for existing in self.questions:
            similarity = SequenceMatcher(None, 
                                      new_question['question'], 
                                      existing['question']).ratio()
            if similarity > self.similarity_threshold:
                return True
        return False
    
    def add_question(self, question_data):
        """Adds new question and checks similarity"""
        question_hash = self.hash_question(question_data['question'])
        
        if question_hash not in self.question_hashes and not self.is_similar(question_data):
            self.questions.append(question_data)
            self.question_hashes.add(question_hash)
            return True
        return False

# Input models
class GenerateQuestionsRequest(BaseModel):
    main_topic: str
    sub_topics: List[str]
    content: str
    question_type: str
    question_count: int
    difficulty: int
    similarity_threshold: float = 0.7
    model: str
    model_temperature: float = 0.3
    use_rag: bool = True  # RAG kullanımını kontrol etmek için
    openai_api_key: str = "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",
    anthropic_api_key: str = "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",
    
from pydantic import BaseModel, Field
from typing import Optional, Literal

class AnalyzeQuestionRequest(BaseModel):
    question_text: str
    model_type: Literal["gpt", "claude"] = Field(
        default="claude",
        description="Select the model type to use for analysis"
    )
    model_name: Optional[str] = Field(
        default="claude-3-5-haiku-20241022",
        description="Specific model name to use"
    )
    openai_api_key: Optional[str] = Field(
        default="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",
        description="OpenAI API key for GPT models"
    )
    anthropic_api_key: Optional[str] = Field(
        default="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",
        description="Anthropic API key for Claude models"
    )
    
class ModelConfig:
    GPT_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-16k","gpt-4","gpt-4-32k"]
    CLAUDE_MODELS = ["claude-3-haiku-20240307", "claude-3-5-haiku-20241022"]
    
    @staticmethod
    def validate_model(model_type: str, model_name: str) -> bool:
        if model_type == "gpt" and model_name in ModelConfig.GPT_MODELS:
            return True
        elif model_type == "claude" and model_name in ModelConfig.CLAUDE_MODELS:
            return True
        return False

class QuestionAnalyzer:
    def __init__(self):
        self.analysis_prompt = """
        Aşağıda bir soru, doğru cevap ve açıklaması verilmiştir. Lütfen sorunun doğruluğunu ve mantıklılığını değerlendirerek 100 üzerinden bir puan verin.

        **Soru:**  
        {question_text}

        **İnceleme Kriterleri:**  
        - Sorunun konuya uygunluğu ve netliği 
        - Doğru cevabın mantıklı olup olmadığı 
        - Açıklamanın yeterli ve anlaşılır olması 

        Lütfen sadece aşağıdaki formatta bir çıktı üretin:
        - "Puan: X/100"
        - "Açıklama: [Kısa ve öz analiz]"
        """

    async def analyze_with_claude(
        self, 
        question_text: str, 
        api_key: str, 
        model_name: str
    ) -> Tuple[int, str]:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = await asyncio.to_thread(
                client.messages.create,
                model=model_name,
                messages=[{
                    "role": "user", 
                    "content": self.analysis_prompt.format(question_text=question_text)
                }],
                temperature=0.3,
                max_tokens=400
            )
            
            response_text = response.content[0].text
            score_match = re.search(r"Puan:\s*(\d+)/100", response_text)
            score = int(score_match.group(1)) if score_match else 0
            
            return score, response_text
        except Exception as e:
            logger.error(f"Claude analysis error: {str(e)}")
            raise

    async def analyze_with_gpt(
        self, 
        question_text: str, 
        api_key: str, 
        model_name: str
    ) -> Tuple[int, str]:
        try:
            client = OpenAI(api_key=api_key)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model_name,
                messages=[{
                    "role": "user", 
                    "content": self.analysis_prompt.format(question_text=question_text)
                }],
                temperature=0.3,
                max_tokens=400
            )
            
            response_text = response.choices[0].message.content
            score_match = re.search(r"Puan:\s*(\d+)/100", response_text)
            score = int(score_match.group(1)) if score_match else 0
            
            return score, response_text
        except Exception as e:
            logger.error(f"GPT analysis error: {str(e)}")
            raise

    async def analyze_question(
        self,
        question_text: str,
        model_type: str,
        model_name: str,
        api_keys: Dict[str, str]
    ) -> Tuple[int, str]:
        if not ModelConfig.validate_model(model_type, model_name):
            raise ValueError(f"Invalid model selection: {model_name} for {model_type}")
        
        if model_type == "claude":
            if not api_keys.get("anthropic_api_key"):
                raise ValueError("Anthropic API key is required for Claude models")
            return await self.analyze_with_claude(
                question_text, 
                api_keys["anthropic_api_key"], 
                model_name
            )
        else:  # gpt
            if not api_keys.get("openai_api_key"):
                raise ValueError("OpenAI API key is required for GPT models")
            return await self.analyze_with_gpt(
                question_text, 
                api_keys["openai_api_key"], 
                model_name
            )
# Response models
class QuestionData(BaseModel):
    question: str
    choices: str
    correct_answer: str
    explanation: str

class GenerateQuestionsResponse(BaseModel):
    success: bool
    total_generated: int
    questions: List[QuestionData]
    message: str

class AnalyzeQuestionResponse(BaseModel):
    score: int
    analysis: str
# Prompt generation functions
def get_difficulty_prompt(difficulty_level: int) -> str:
    """Generate appropriate difficulty-specific prompt based on the level"""
    
    if difficulty_level in DIFFICULTY_LEVELS["BEGINNER"]:
        return """
        ZORLUK SEVİYESİ KURALLARI (%10-%50 - KOLAY SEVİYE):
        1. Soru Yapısı:
           - Temel kavramları soran sorular
           - Tek bir kavram veya tanım üzerine odaklanan sorular
           - Doğrudan hatırlama gerektiren sorular
        
        2. Beklenen Bilgi Düzeyi:
           - Temel terminoloji
           - Basit tanımlar
           - Açık ve net kavramlar
        
        3. Cevaplama Süresi: 1-2 dakika
        
        4. Düşünme Seviyesi:
           - Hatırlama
           - Tanımlama
           - Basit uygulama
        """
    
    elif difficulty_level in DIFFICULTY_LEVELS["INTERMEDIATE"]:
        return """
        ZORLUK SEVİYESİ KURALLARI (%51-%80 - ORTA SEVİYE):
        1. Soru Yapısı:
           - Birden fazla kavramı birleştiren sorular
           - Analiz gerektiren durumlar
           - Karşılaştırma yapılması gereken sorular
        
        2. Beklenen Bilgi Düzeyi:
           - Kavramlar arası ilişkiler
           - Teorik bilgilerin uygulanması
           - Orta düzey analiz yeteneği
        
        3. Cevaplama Süresi: 2-4 dakika
        
        4. Düşünme Seviyesi:
           - Analiz
           - Uygulama
           - Karşılaştırma
           - Temel problem çözme
        """
    
    else:  # ADVANCED
        return """
        ZORLUK SEVİYESİ KURALLARI (%81-%100 - ZOR/AKADEMİK SEVİYE):
        1. Soru Yapısı:
           - Kompleks akademik analiz gerektiren sorular
           - Çoklu kavram ve teorilerin sentezi
           - İleri düzey problem çözme
           - Akademik araştırma gerektiren konular
           - Literatür Taraması Gerektiren konular
        
        2. Zorunlu İçerik Gereklilikleri:
           - En az 3 farklı akademik kaynak referansı
           - Güncel akademik tartışmalara atıf
           - İleri düzey teorik çerçeve
           - Güncel kaynaklara atıf
        
        3. Hukuki Sorular İçin Özel Gereklilikler:
           - En az 2 farklı kanun maddesi referansı
           - Güncel içtihatlardan örnekler
           - Doktrinel tartışmalara atıf
           - Karşılaştırmalı hukuk analizi
           - Herzaman anayasa ve hukuk maddelerine deyinilmeli deyinilmeden soru sorulmamalı
        
        4. Cevaplama Süresi: 10-15 dakika
        
        5. Düşünme Seviyesi:
           - İleri düzey analiz
           - Sentez
           - Değerlendirme
           - Akademik araştırma
           - Kompleks problem çözme
        6. Zorunlu Durumlar
            - Asla kolay kabul edilebilir sorular sorulamaz
            - Asla tek düze sığ çerçevede sorular sorulamaz
            - Asla basit kavramlar sorulamaz
        """

def validate_difficulty(difficulty_level: int) -> bool:
    """Validate if the given difficulty level is within acceptable ranges"""
    return any(difficulty_level in level_range for level_range in DIFFICULTY_LEVELS.values())

# Prompt generation functions
def generate_multiple_questions_prompt(main_topic, sub_topics, content, question_type, generated_count, total_count, difficulty):
    if not validate_difficulty(difficulty):
        raise ValueError(f"Geçersiz zorluk seviyesi: {difficulty}. Zorluk seviyesi 10-100 arasında olmalıdır.")
    
    difficulty_guidelines = get_difficulty_prompt(difficulty)
    """Enhanced prompt for generating unique questions"""
    prompt = f"""   
    Lütfen aşağıdaki bilgileri kullanarak {total_count} sorudan {generated_count+1}. soruyu Türkçe oluştur:
    Ana Konu: {main_topic}
    Alt Başlıklar: {', '.join(sub_topics)}
    İçerik: {content}
    Soru Türü: {question_type}
    Zorluk Seviyesi: %{difficulty}

    {difficulty_guidelines}
       
    Önemli Kurallar:
    1. Şu ana kadar {generated_count} adet soru üretildi, şimdi TAM OLARAK BENZERSİZ bir {generated_count+1}. soru üretmelisin.
    2. Her soru birbirinden tamamen farklı olmalıdır - farklı alt konulara odaklanmalıdır.
    3. Daha önce üretilen sorulardan TAMAMEN farklı bir soru üret.
    4. Her soru için aşağıdaki formatı kullan:
    5. Her Çoktan Seçmeli soru için her zaman 5 Şık koymalısın A,B,C,D,E Şıkları HERZAMAN OLMALI
    6. Her Doğru Yanlış sorusunda HERZAMAN İKİ ŞIK OLACAK A)Doğru, B)Yanlış Şıkları HERZAMAN OLMALI
    7. Her Açık Uçlu soru için HİÇBİR ZAMAN ŞIK KULLANMAYACAKSIN AMA HERZAMAN SORUYU YAZMALISIN AÇIKLAMASI İLE BİRLİKTE  ŞIKLAR ASLA OLMAMALI AÇIK UÇLU SORULARDA
    8. Her Açık Uçlu soru için Herzaman Soru Yazmalısın Soru Yazmama ihtimalin YOK.
    9. Her şartta hangi srou türü olursa olsun HERZAMAN SORU YAZMALISIN VE ŞU FORMATTA YAZMALISIN Soru: [Soru Metni] ŞEKLİNDE YAZACAKSIN
    10. Her şartta ürettiğin sorular herzaman zorluk değeri olarak %{difficulty} değerine uygun olmalıdır
    11. Soru Türü: Doğru/Yanlış olan Sorularda SORU İÇERİĞİ DOĞRU YANLIŞ CEVABINA UYGUN ŞEKİLDE SORULMALIDIR. SORUNUN AKIŞI İÇERİĞİ VE SORU KALIBI DOĞRU YANLIŞ SORUSUNA UYGUN OLMALIDIR
    12. Soru Türü: Doğru/Yanlış olan sorularda SORU HER ZAMAN BİRŞEYİN DURUMUNUN DOĞRULUĞUNU YADA YANLIŞLIĞINI SORUYOR OLMALI ANCAK SORUNUN AKIŞIDA, SORUNUN KALIBIDA, SORUNUN İÇERİĞİDE DOĞRU YANLIŞ SORUSUNA UYGUN OLARAK YAZILMALIDIR HER ŞARTTA Doğru/Yanlış Sorularında soru DOĞRU VE YANLIŞ OLARAK CEVAPLANABİLİR ŞEKİLDE SORULMALIDIR. SORUNUN İÇERİĞİ,AKIŞI VE SORU KALIBI SORUNUN DOĞRU YADA YANLIŞ OLARAK CEVAPLANABİLMESİNE UYGUN OLMALIDIR 
    Soru: [Soru metni]
    Şıklar: (Eğer çoktan seçmeliyse)
    A) ...
    B) ...
    C) ...
    D) ...
    E) ...
    
    Doğru Cevap: [Cevap]
    
    Açıklama: [Detaylı açıklama]
    
    ZORUNLU KONTROLLER:
    - Tamamen farklı bir konsepte odaklanmalısın
    - Aynı kavramları sormaktan kaçınmalısın
    - Şıklar ve açıklamalar benzersiz olmalıdır
    - Her yeni soru öncekilerden belirgin şekilde farklı olmalıdır
    - Soru tam olarak %{difficulty} zorluk seviyesinde olmalıdır
    - Doğru/Yanlış Sorularında SORU İÇERİĞİ DOĞRU YANLIŞ CEVABINA UYGUN ŞEKİLDE SORULMALIDIR
    - Doğru/Yanlış sorularında SORU HER ZAMAN BİRŞEYİN DURUMUNUN DOĞRULUĞUNU YADA YANLIŞLIĞINI SORUYOR OLMALI HER ŞARTTA Doğru/Yanlış Sorularında soru DOĞRU VE YANLIŞ OLARAK CEVAPLANABİLİR ŞEKİLDE SORULMALIDIR
    - Soru Türü: Doğru/Yanlış olan Sorularda SORU İÇERİĞİ DOĞRU YANLIŞ CEVABINA UYGUN ŞEKİLDE SORULMALIDIR. SORUNUN AKIŞI İÇERİĞİ VE SORU KALIBI DOĞRU YANLIŞ SORUSUNA UYGUN OLMALIDIR
    - Soru Türü: Doğru/Yanlış olan sorularda SORU HER ZAMAN BİRŞEYİN DURUMUNUN DOĞRULUĞUNU YADA YANLIŞLIĞINI SORUYOR OLMALI ANCAK SORUNUN AKIŞIDA, SORUNUN KALIBIDA, SORUNUN İÇERİĞİDE DOĞRU YANLIŞ SORUSUNA UYGUN OLARAK YAZILMALIDIR HER ŞARTTA Doğru/Yanlış Sorularında soru DOĞRU VE YANLIŞ OLARAK CEVAPLANABİLİR ŞEKİLDE SORULMALIDIR. SORUNUN İÇERİĞİ,AKIŞI VE SORU KALIBI SORUNUN DOĞRU YADA YANLIŞ OLARAK CEVAPLANABİLMESİNE UYGUN OLMALIDIR.          
    """
    return prompt

def generate_prompt(main_topic, sub_topics, content, question_type, question_count, difficulty):
    """Original prompt function preserved"""
    
    if not validate_difficulty(difficulty):
        raise ValueError(f"Geçersiz zorluk seviyesi: {difficulty}. Zorluk seviyesi 10-100 arasında olmalıdır.")
    
    difficulty_guidelines = get_difficulty_prompt(difficulty)
    
    
    prompt = f"""
    Lütfen aşağıdaki bilgileri kullanarak Türkçe soru oluştur:\n
    Ana Konu: {main_topic}\n
    Alt Başlıklar: {', '.join(sub_topics)}\n
    İçerik: {content}\n
    Soru Türü: {question_type}\n
    Zorluk Seviyesi: %{difficulty}\n

    {difficulty_guidelines}

    Aşağıdaki kurallara mutlaka uyunuz:\n
    1. Sorunun başına her zaman "Soru:" ekleyin.\n
    2. Eğer şıklar varsa, "Şıklar:" ifadesi ile her bir şıkkı alt alta sıralayın.\n
    3. Şıkların ardından "Doğru Cevap:" ifadesiyle doğru cevabı belirtin.\n
    4. En sona "Açıklama:" ekleyerek doğru cevabın neden doğru olduğunu açıklayın.\n
    5. Şıklar arasında ve diğer bölümler arasında **her zaman birer boş satır bırakın.**\n
    6. Her bir şık ayrı bir satırda yer almalıdır.\n
    7. Her bir şıkdan sonra **her zaman birer boş satır bırakın.**\n
    8. Her başlık ve içeriğinden sonra **her zaman birer boş satır bırakın.**\n
    9. Her "Açıklama:" yazımında sonra **her zaman bir boş satır bırakın.**.\n
    10.Her Çoktan Seçmeli soru için her zaman 5 Şık koymalısın A,B,C,D,E Şıkları HERZAMAN OLMALI
    11.Her Doğru/Yanlış Sorusunda HERZAMAN 2 ŞIK OLMALI A)Doğru, B)Yanlış şıkları HERZAMAN OLMALI
    12.Her Açık Uçlu soru için HİÇBİR ZAMAN ŞIK KULLANMAYACAKSIN AMA HERZAMAN SORUYU YAZMALISIN AÇIKLAMASI İLE BİRLİKTE ŞIKLAR ASLA OLMAMALI AÇIK UÇLU SORULARDA
    13.Her Açık Uçlu soru için Herzaman Soru Yazmalısın Soru Yazmama ihtimalin YOK.
    14.Her şartta ürettiğin sorular herzaman zorluk değeri olarak %{difficulty} değerine uygun olmalıdır
    15.Doğru/Yanlış Sorularında SORU İÇERİĞİ DOĞRU YANLIŞ CEVABINA UYGUN ŞEKİLDE SORULMALIDIR
    16.Doğru/Yanlış sorularında SORU HER ZAMAN BİRŞEYİN DURUMUNUN DOĞRULUĞUNU YADA YANLIŞLIĞINI SORUYOR OLMALI HER ŞARTTA Doğru/Yanlış Sorularında soru DOĞRU VE YANLIŞ OLARAK CEVAPLANABİLİR ŞEKİLDE SORULMALIDIR
    17.Soru Türü: Doğru/Yanlış olan Sorularda SORU İÇERİĞİ DOĞRU YANLIŞ CEVABINA UYGUN ŞEKİLDE SORULMALIDIR. SORUNUN AKIŞI İÇERİĞİ VE SORU KALIBI DOĞRU YANLIŞ SORUSUNA UYGUN OLMALIDIR
    18.Soru Türü: Doğru/Yanlış olan sorularda SORU HER ZAMAN BİRŞEYİN DURUMUNUN DOĞRULUĞUNU YADA YANLIŞLIĞINI SORUYOR OLMALI ANCAK SORUNUN AKIŞIDA, SORUNUN KALIBIDA, SORUNUN İÇERİĞİDE DOĞRU YANLI�� SORUSUNA UYGUN OLARAK YAZILMALIDIR HER ŞARTTA Doğru/Yanlış Sorularında soru DOĞRU VE YANLIŞ OLARAK CEVAPLANABİLİR ŞEKİLDE SORULMALIDIR. SORUNUN İÇERİĞİ,AKIŞI VE SORU KALIBI SORUNUN DOĞRU YADA YANLIŞ OLARAK CEVAPLANABİLMESİNE UYGUN OLMALIDIR.        
   **Önemli Notlar:**
    - Her Çoktan Seçmeli soru için her zaman 5 Şık koymalısın A,B,C,D,E Şıkları HERZAMAN OLMALI
    - Her Doğru/Yanlış Sorusunda HERZAMAN 2 ŞIK OLMALI A)Doğru, B)Yanlış şıkları HERZAMAN OLMALI
    - Her Açık Uçlu soru için HİÇBİR ZAMAN ŞIK KULLANMAYACAKSIN AMA HERZAMAN SORUYU YAZMALISIN AÇIKLAMASI İLE BİRLİKTE  ŞIKLAR ASLA OLMAMALI AÇIK UÇLU SORULARDA
    - HER ZAMAN ürettiğin sorular zorluk değeri olarak %{difficulty} değerine UYGUN OLAMLIDIR.
    - Üretilen sorular, içerik ve şıklar bakımından büyük farklılıklar içermelidir.
    - Herzaman 1 adet Soru: başlıklı soru üreteceksin asla birden fazla soru üretemezsin.
    - Üretilen sorular her şartta herzaman birbirinden farklı akış anlam ve içeriklere sahip olmalıdır.
    - Doğru/Yanlış Sorularında SORU İÇERİĞİ DOĞRU YANLIŞ CEVABINA UYGUN ŞEKİLDE SORULMALIDIR
    - Doğru/Yanlış sorularında SORU HER ZAMAN BİRŞEYİN DURUMUNUN DOĞRULUĞUNU YADA YANLIŞLIĞINI SORUYOR OLMALI HER ŞARTTA Doğru/Yanlış Sorularında soru DOĞRU VE YANLIŞ OLARAK CEVAPLANABİLİR ŞEKİLDE SORULMALIDIR
    - Soru Türü: Doğru/Yanlış olan Sorularda SORU İÇERİĞİ DOĞRU YANLIŞ CEVABINA UYGUN ŞEKİLDE SORULMALIDIR. SORUNUN AKIŞI İÇERİĞİ VE SORU KALIBI DOĞRU YANLIŞ SORUSUNA UYGUN OLMALIDIR
    - Soru Türü: Doğru/Yanlış olan sorularda SORU HER ZAMAN BİRŞEYİN DURUMUNUN DOĞRULUĞUNU YADA YANLIŞLIĞINI SORUYOR OLMALI ANCAK SORUNUN AKIŞIDA, SORUNUN KALIBIDA, SORUNUN İÇERİĞİDE DOĞRU YANLIŞ SORUSUNA UYGUN OLARAK YAZILMALIDIR HER ŞARTTA Doğru/Yanlış Sorularında soru DOĞRU VE YANLIŞ OLARAK CEVAPLANABİLİR ŞEKİLDE SORULMALIDIR. SORUNUN İÇERİĞİ,AKIŞI VE SORU KALIBI SORUNUN DOĞRU YADA YANLIŞ OLARAK CEVAPLANABİLMESİNE UYGUN OLMALIDIR.      
    """
    return prompt
def extract_question_details(response_text):
    """Extracts question, choices, correct answer and explanation from the response text."""
    question_match = re.search(r"Soru:\s*(.*?)\n\n", response_text, re.DOTALL)
    choices_match = re.findall(r"([A-E])\)\s*(.*?)\n", response_text)
    answer_match = re.search(r"Doğru Cevap:\s*(.*?)\n\n", response_text, re.DOTALL)
    explanation_match = re.search(r"Açıklama:\s*(.*)", response_text, re.DOTALL)

    question = question_match.group(1).strip() if question_match else "Soru bulunamadı."
    choices = "\n".join([f"{choice[0]}) {choice[1]}" for choice in choices_match]) if choices_match else "Şıklar bulunamadı."
    correct_answer = answer_match.group(1).strip() if answer_match else "Doğru cevap bulunamadı."
    explanation = explanation_match.group(1).strip() if explanation_match else "Açıklama bulunamadı."
    
    return question, choices, correct_answer, explanation

async def generate_llm_response(model_params, model_type, api_key, prompt, question_tracker, rag_manager):
    """RAG destekli LLM yanıt üretme fonksiyonu"""
    max_regeneration_attempts = 2
    current_attempt = 0
    relevant_contexts = []
    
    while current_attempt < max_regeneration_attempts:
        try:
            # Get RAG context if available
            if rag_manager:
                relevant_contexts = rag_manager.get_relevant_context(prompt)
                context_text = '\n'.join(relevant_contexts) if relevant_contexts else ""
                
                enhanced_prompt = f"""
                KONU İLE İLGİLİ KAYNAKLAR:
                {context_text}

                ÖNEMLİ TALİMATLAR:
                1. Bu kaynakları SADECE ana konuyla doğrudan ilgili olduğunda kullan
                2. Kaynakların içeriğini olduğu gibi kullanma, konuya uygun şekilde dönüştür
                3. Her zaman asıl soru talebine ve konuya odaklan
                4. Kaynakları sadece destekleyici bilgi olarak kullan

                ASIL SORU OLUŞTURMA GÖREVİ:
                {prompt}
                """
                prompt = enhanced_prompt
            
            # Generate response based on model type
            if model_type == "openai":
                client = OpenAI(api_key=api_key)
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model_params["model"],
                    messages=[{
                        "role": "system",
                        "content": "Sen bir eğitim uzmanısın. Sadece verilen konu ve içerikle ilgili sorular üret."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=model_params["temperature"],
                    max_tokens=4096
                )
                response_message = response.choices[0].message.content
                
            elif model_type == "anthropic":
                client = anthropic.Anthropic(api_key=api_key)
                response = await asyncio.to_thread(
                    client.messages.create,
                    model=model_params["model"],
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=model_params["temperature"],
                    max_tokens=4096
                )
                response_message = response.content[0].text
            
            # Extract question details
            question, choices, correct_answer, explanation = extract_question_details(response_message)
            
            # Create question data
            question_data = {
                "question": question,
                "choices": choices,
                "correct_answer": correct_answer,
                "explanation": explanation,
                "sources": relevant_contexts
            }
            
            # Check if question is unique and relevant
            if question_tracker.add_question(question_data):
                return True, question_data, None
            
            current_attempt += 1
            if "temperature" in model_params:
                model_params["temperature"] = min(1.0, model_params["temperature"] + 0.1)
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return False, None, f"API error: {str(e)}"
            
    return False, None, "Maksimum yeniden oluşturma denemesi aşıldı."

async def analyze_question_with_claude(question_text: str, anthropic_api_key: str, model: str = "claude-3-sonnet-20240229"):
    """Analyzes the question using Claude for accuracy and logic"""
    # Validate model selection
    valid_models = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
        "gpt-4o"
    ]
    
    if model not in valid_models:
        raise ValueError(f"Invalid model selection. Please choose from: {', '.join(valid_models)}")

    prompt = f""" 
    Aşağıda bir soru, doğru cevap ve açıklaması verilmiştir. Lütfen sorunun doğruluğunu ve mantıklılığını değerlendirerek 100 üzerinden bir puan verin.

    **Soru:**  
    {question_text}

    **İnceleme Kriterleri:**  
    - Sorunun konuya uygunluğu ve netliği 
    - Doğru cevabın mantıklı olup olmadığı 
    - Açıklamanın yeterli ve anlaşılır olması 

    Lütfen sadece aşağıdaki formatta bir çıktı üretin:
    - "Puan: X/100"
    - "Açıklama: [Kısa ve öz analiz]"
    """

    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        response = await asyncio.to_thread(
            client.messages.create,
            model=model,  # Use the provided model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )

        response_text = response.content[0].text

        # Extract score using regex
        score_match = re.search(r"Puan:\s*(\d+)/100", response_text)
        score = int(score_match.group(1)) if score_match else 0

        return score, response_text
    except Exception as e:
        logger.error(f"Error analyzing question: {str(e)}")
        return 0, f"Analiz sırasında hata oluştu: {str(e)}"



async def generate_regeneration_prompt(request: RegenerateQuestionRequest) -> str:
    """Generate a prompt for question regeneration"""
    difficulty_guidelines = get_difficulty_prompt(request.difficulty)
    
    prompt = f"""
    Lütfen aşağıdaki orijinal soruyu ve bilgileri kullanarak YENİ ve BENZER bir soru oluşturun:

    Orijinal Soru:
    {request.original_question}

    Ana Konu: {request.main_topic}
    Alt Başlıklar: {', '.join(request.sub_topics)}
    İçerik: {request.content}
    Soru Türü: {request.question_type}
    Zorluk Seviyesi: %{request.difficulty}

    {difficulty_guidelines}

    Yeni soru için özel talimatlar:
    1. Orijinal sorunun ana fikrini koruyun ancak farklı bir yaklaşım kullanın
    2. Zorluk seviyesi aynı kalmalıdır (%{request.difficulty})
    3. Soru türü aynı kalmalıdır
    4. Her Çoktan Seçmeli soru için her zaman 5 Şık koymalısın A,B,C,D,E Şıkları HERZAMAN OLMALI
    5. Her Doğru Yanlış sorusunda HERZAMAN İKİ ŞIK OLMALI A)Doğru, B)Yanlış şıkları HERZAMAN OLMALI
    6. Her Açık Uçlu soru için HİÇBİR ZAMAN ŞIK KULLANMAYACAKSIN AMA HERZAMAN SORUYU YAZMALISIN AÇIKLAMASI İLE BİRLİKTE

    Lütfen aşağıdaki formatta yeni bir soru oluşturun:
    Soru: [Yeni soru metni]
    Şıklar: (Eğer çoktan seçmeliyse)
    A) ...
    B) ...
    C) ...
    D) ...
    E) ...
    
    Doğru Cevap: [Cevap]
    
    Açıklama: [Detaylı açıklama]
    """
    return prompt

@app.post("/regenerate-question", response_model=RegenerateQuestionResponse)
async def regenerate_question(request: RegenerateQuestionRequest):
    try:
        # API key validation
        if (request.model.startswith("gpt") and not request.openai_api_key) or \
           (request.model.startswith("gemini") and not request.google_api_key) or \
           (request.model.startswith("claude") and not request.anthropic_api_key):
            raise HTTPException(status_code=400, detail="Selected model requires appropriate API key")
        
        # Model type determination
        model_type = None
        api_key = None
        if request.model.startswith("gpt"):
            model_type = "openai"
            api_key = request.openai_api_key
        elif request.model.startswith("gemini"):
            model_type = "google"
            api_key = request.google_api_key
        elif request.model.startswith("claude"):
            model_type = "anthropic"
            api_key = request.anthropic_api_key
        
        # Generate regeneration prompt
        prompt = await generate_regeneration_prompt(request)
        
        # Initialize question tracker
        question_tracker = QuestionTracker(similarity_threshold=request.similarity_threshold)
        
        # Add original question to tracker
        original_question_data = {
            "question": request.original_question,
            "choices": "",
            "correct_answer": "",
            "explanation": ""
        }
        question_tracker.add_question(original_question_data)
        
        # Generate new question
        model_params = {
            "model": request.model,
            "temperature": request.model_temperature
        }
        
        max_attempts = 5
        current_attempt = 0
        
        while current_attempt < max_attempts:
            try:
                success, question_data, error = await generate_llm_response(
                    model_params=model_params,
                    model_type=model_type,
                    api_key=api_key,
                    prompt=prompt,
                    question_tracker=question_tracker
                )
                
                if success and question_data:
                    return RegenerateQuestionResponse(
                        success=True,
                        original_question=request.original_question,
                        question=question_data["question"],
                        choices=question_data["choices"],
                        correct_answer=question_data["correct_answer"],
                        explanation=question_data["explanation"],
                        message="Soru başarıyla yeniden oluşturuldu!"
                    )
                
                current_attempt += 1
                if "temperature" in model_params:
                    model_params["temperature"] = min(1.0, model_params["temperature"] + 0.1)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in regeneration attempt {current_attempt}: {str(e)}")
                current_attempt += 1
                await asyncio.sleep(1)
        
        raise HTTPException(
            status_code=500,
            detail="Maximum regeneration attempts exceeded without producing a suitable question"
        )
        
    except Exception as e:
        logger.error(f"Error in question regeneration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to regenerate question: {str(e)}"
        )
# Rate limiter class ekleyelim
# Rate limiter class ekleyelim
class RateLimiter:
    def __init__(self, requests_per_minute=10000):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            now = time.time()
            # 60 saniyeden eski istekleri temizle
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            if len(self.request_times) >= self.requests_per_minute:
                # En eski istek zamanını al
                oldest_request = self.request_times[0]
                # Bir sonraki istek için beklenecek süre
                
                await asyncio.sleep(0.1)
            
            self.request_times.append(now)

@app.post("/generate-questions", response_model=GenerateQuestionsResponse)
async def generate_questions(request: GenerateQuestionsRequest):
    try:
        # API key validation
        if (request.model.startswith("gpt") and not request.openai_api_key) or \
           (request.model.startswith("claude") and not request.anthropic_api_key):
            raise HTTPException(status_code=400, detail="Selected model requires appropriate API key")
        
        # Model type determination
        model_type = None
        api_key = None
        if request.model.startswith("gpt"):
            model_type = "openai"
            api_key = request.openai_api_key
        elif request.model.startswith("claude"):
            model_type = "anthropic"
            api_key = request.anthropic_api_key
        
        # Rate limiter initialization
        rate_limiter = RateLimiter()
        
        model_params = {
            "model": request.model,
            "temperature": request.model_temperature,
        }
        
        question_tracker = QuestionTracker(similarity_threshold=request.similarity_threshold)
        
        async def generate_single_question(index: int):
            try:    
                await rate_limiter.acquire()
                
                current_params = model_params.copy()
                current_params["temperature"] = min(1.0, model_params["temperature"] + (index * 0.05))
                
                prompt = generate_multiple_questions_prompt(
                    request.main_topic,
                    request.sub_topics,
                    request.content,
                    request.question_type,
                    index,
                    request.question_count,
                    request.difficulty
                ) if index > 0 else generate_prompt(
                    request.main_topic,
                    request.sub_topics,
                    request.content,
                    request.question_type,
                    request.question_count,
                    request.difficulty
                )
                
                success, question_data, error = await generate_llm_response(
                    model_params=current_params,
                    model_type=model_type,
                    api_key=api_key,
                    prompt=prompt,
                    question_tracker=question_tracker,
                    rag_manager=rag_manager if request.use_rag else None
                )
                
                if success:
                    return success, question_data, error
                
                return False, None, "Soru üretilemedi"
                
            except Exception as e:
                logger.error(f"Error in question generation {index}: {str(e)}")
                return False, None, str(e)
        
        # Soru üretme işlemi
        successful_questions = []
        batch_size = 3
        total_attempts = 0
        max_attempts = request.question_count * 2
        
        while len(successful_questions) < request.question_count and total_attempts < max_attempts:
            remaining = request.question_count - len(successful_questions)
            current_batch_size = min(batch_size, remaining)
            
            batch_tasks = [
                generate_single_question(len(successful_questions) + i)
                for i in range(current_batch_size)
            ]
            
            try:
                if total_attempts > 0:
                    await asyncio.sleep(0.5)
                
                batch_results = await asyncio.gather(*batch_tasks)
                
                for success, question_data, error in batch_results:
                    if success and question_data:
                        successful_questions.append(QuestionData(
                            question=question_data["question"],
                            choices=question_data["choices"],
                            correct_answer=question_data["correct_answer"],
                            explanation=question_data["explanation"]
                        ))
                        
                        if len(successful_questions) >= request.question_count:
                            break
                
                total_attempts += current_batch_size
                
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                if len(successful_questions) == 0:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Failed to generate questions: {str(e)}"
                    )
        
        # Prepare response
        success = len(successful_questions) > 0
        message = (
            f"{len(successful_questions)} soru başarıyla oluşturuldu!"
            if len(successful_questions) >= request.question_count
            else f"{len(successful_questions)}/{request.question_count} soru oluşturulabildi."
        )
        
        return GenerateQuestionsResponse(
            success=success,
            total_generated=len(successful_questions),
            questions=successful_questions,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Error in question generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate questions: {str(e)}"
        )   
question_analyzer = QuestionAnalyzer()

@app.post("/analyze-question")
async def analyze_question(request: AnalyzeQuestionRequest):
    try:
        # Validate the model selection
        if not request.model_name:
            if request.model_type == "claude":
                request.model_name = "claude-3-sonnet-20240229"
            else:
                request.model_name = "gpt-4o"
        
        # Prepare API keys
        api_keys = {
            "openai_api_key": request.openai_api_key,
            "anthropic_api_key": request.anthropic_api_key
        }
        
        # Analyze the question
        score, analysis = await question_analyzer.analyze_question(
            question_text=request.question_text,
            model_type=request.model_type,
            model_name=request.model_name,
            api_keys=api_keys
        )
        
        return {
            "score": score,
            "analysis": analysis,
            "model_used": f"{request.model_type} ({request.model_name})"
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in question analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze question: {str(e)}")

@app.get("/available-models")
async def get_available_models():
    return {
        "gpt_models": ModelConfig.GPT_MODELS,
        "claude_models": ModelConfig.CLAUDE_MODELS
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)