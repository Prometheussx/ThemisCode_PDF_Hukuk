from typing import List
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import io
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        PDF işleme sınıfı başlatıcısı
        
        Args:
            chunk_size (int): Her bir metin parçasının maksimum boyutu
            chunk_overlap (int): Parçalar arası örtüşme miktarı
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "],
        )
    
    def extract_text_from_pdf(self, pdf_content) -> str:
        """
        PDF içeriğinden metin çıkarma
        
        Args:
            pdf_content: PDF içeriği (bytes veya file-like object)
            
        Returns:
            str: PDF'den çıkarılan metin
        """
        try:
            # Geçici bir dosya oluştur
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                # Eğer bytes ise doğrudan yaz
                if isinstance(pdf_content, bytes):
                    temp_file.write(pdf_content)
                # Eğer file-like object ise read() ile oku ve yaz
                else:
                    temp_file.write(pdf_content.read())
                temp_file.flush()
                
                # PDF'den metin çıkar
                text = extract_text(temp_file.name)
                
            # Geçici dosyayı sil
            os.unlink(temp_file.name)
            
            if not text.strip():
                raise ValueError("PDF'den metin çıkarılamadı veya PDF boş")
                
            return text
            
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {str(e)}")
            raise Exception(f"PDF işleme hatası: {str(e)}")
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Metni küçük parçalara bölme
        
        Args:
            text (str): Bölünecek metin
            
        Returns:
            List[str]: Metin parçaları listesi
        """
        try:
            chunks = self.text_splitter.split_text(text)
            if not chunks:
                raise ValueError("Metin parçalara bölünemedi veya metin boş")
            return chunks
        except Exception as e:
            logger.error(f"Metin bölme hatası: {str(e)}")
            raise Exception(f"Metin bölme hatası: {str(e)}")
    
    
    def process_pdf(self, pdf_content: bytes) -> List[str]:
        try:
            # Convert bytes to a file-like object
            pdf_file_like = io.BytesIO(pdf_content)
            
            # Extract text from PDF
            text = extract_text(pdf_file_like)
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Adjust chunk size for better performance
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)
            
            return chunks
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise