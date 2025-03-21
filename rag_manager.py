from typing import List, Dict
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGManager:
    def __init__(self, persist_directory="./chroma_db"):
        """
        RAG (Retrieval Augmented Generation) yönetici sınıfı
        
        Args:
            persist_directory (str): Vektör veritabanı için kalıcı depolama dizini
        """
        try:
            # Ensure the persist directory exists
            os.makedirs(persist_directory, exist_ok=True)
            
            # Yeni ChromaDB client oluşturma yöntemi
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Transformer modelini ve tokenizer'ı yükle
            model_name = 'distilbert-base-multilingual-cased'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Koleksiyonu oluştur veya var olanı al
            try:
                self.collection = self.client.get_collection(name="hukuk_dokumanlari")
                logger.info("Mevcut koleksiyon kullanılıyor: hukuk_dokumanlari")
            except:
                self.collection = self.client.create_collection(
                    name="hukuk_dokumanlari",
                    metadata={"description": "Hukuk dökümanları için vektör veritabanı"}
                )
                logger.info("Yeni koleksiyon oluşturuldu: hukuk_dokumanlari")
            
            logger.info("RAGManager başarıyla başlatıldı.")
            
        except Exception as e:
            logger.error(f"RAGManager başlatma hatası: {str(e)}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Metinler için embedding vektörleri oluştur
        
        Args:
            texts (List[str]): Embedding'i oluşturulacak metinler listesi
            
        Returns:
            List[List[float]]: Embedding vektörleri listesi
        """
        try:
            # Metinleri tokenize et
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Model çıktısını al
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Son katman çıktısının ortalamasını al
            embeddings = torch.mean(model_output.last_hidden_state, dim=1)
            
            return embeddings.numpy().tolist()
            
        except Exception as e:
            logger.error(f"Embedding oluşturma hatası: {str(e)}")
            raise
    
    def add_documents(self, chunks: List[str], metadata: List[Dict] = None):
        """
        Döküman parçalarını veritabanına ekle
        
        Args:
            chunks (List[str]): Eklenecek metin parçaları
            metadata (List[Dict], optional): Her parça için metadata
        """
        try:
            # Embeddingler oluştur
            embeddings = self.get_embeddings(chunks)
            
            # Metadata yoksa varsayılan oluştur
            if metadata is None:
                metadata = [{"source": "pdf", "chunk_id": str(i)} for i in range(len(chunks))]
            
            # Koleksiyona ekle
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata,
                ids=[f"doc_{i}" for i in range(len(chunks))]
            )
            
            logger.info(f"{len(chunks)} belge parçası başarıyla eklendi.")
            
        except Exception as e:
            logger.error(f"Belge ekleme hatası: {str(e)}")
            raise
    
    def get_relevant_context(self, query: str, n_results: int = 5) -> List[str]:
        """
        Sorgu için en alakalı içeriği getir
        
        Args:
            query (str): Arama sorgusu
            n_results (int): Dönülecek sonuç sayısı
            
        Returns:
            List[str]: İlgili döküman parçaları
        """
        try:
            # Sorgu için embedding oluştur
            query_embedding = self.get_embeddings([query])[0]
            
            # En yakın dökümanları bul
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            logger.info(f"Sorgu için {n_results} sonuç bulundu.")
            return results["documents"][0]
            
        except Exception as e:
            logger.error(f"İçerik getirme hatası: {str(e)}")
            raise

    def clear_collection(self):
        """Koleksiyondaki tüm verileri temizle"""
        try:
            self.collection.delete(ids=self.collection.get()["ids"])
            logger.info("Koleksiyon başarıyla temizlendi.")
        except Exception as e:
            logger.error(f"Koleksiyon temizleme hatası: {str(e)}")
            raise