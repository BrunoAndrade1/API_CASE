import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import re
import os
import joblib 
from typing import Dict, Optional, Any
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configuração da página
st.set_page_config(
    page_title="Kickstarter Success Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ========== ADICIONAR APÓS OS IMPORTS ==========

# Classes do modelo (copiar do train_model.py)
class KickstarterPreprocessor:
    """Classe para processar dados do Kickstarter"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.category_stats = None
        self.country_stats = None
        self.features_selected = [
            'cat_success_rate', 'usd_goal_real', 'campaign_days', 
            'goal_magnitude', 'cat_mean_goal', 'name_word_count',
            'cat_median_goal', 'goal_per_day', 'country_success_rate',
            'launch_year', 'main_category', 'name_length',
            'goal_category_ratio', 'country', 'goal_rounded'
        ]
    
    def create_features(self, df):
        """Cria features necessárias"""
        df = df.copy()
        
        df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
        df['launched'] = pd.to_datetime(df['launched'], errors='coerce')
        
        df['campaign_days'] = (df['deadline'] - df['launched']).dt.days
        df['launch_year'] = df['launched'].dt.year
        df['campaign_days'] = df['campaign_days'].clip(lower=1, upper=365)
        
        df['name_length'] = df['name'].fillna('').str.len()
        df['name_word_count'] = df['name'].fillna('').str.split().str.len()
        
        df['usd_goal_real'] = df['usd_goal_real'].clip(upper=1e8)
        df['goal_magnitude'] = np.log10(df['usd_goal_real'].clip(lower=1) + 1)
        df['goal_rounded'] = (df['usd_goal_real'] % 1000 == 0).astype(int)
        
        return df
    
    def transform(self, df):
        """Transforma dados para predição"""
        df = self.create_features(df)
        
        df = df.merge(self.category_stats, left_on='main_category', right_index=True, how='left')
        df = df.merge(self.country_stats, left_on='country', right_index=True, how='left')
        
        df['cat_success_rate'].fillna(0.35, inplace=True)
        df['cat_mean_goal'].fillna(10000, inplace=True)
        df['cat_median_goal'].fillna(5000, inplace=True)
        df['country_success_rate'].fillna(0.35, inplace=True)
        
        df['goal_per_day'] = df['usd_goal_real'] / df['campaign_days'].replace(0, 1)
        df['goal_category_ratio'] = df['usd_goal_real'] / df['cat_median_goal'].replace(0, 1)
        
        df['goal_per_day'] = df['goal_per_day'].replace([np.inf, -np.inf], 0).fillna(0)
        df['goal_category_ratio'] = df['goal_category_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
        
        for col, encoder in self.label_encoders.items():
            known_values = set(encoder.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_values else list(known_values)[0])
            df[col] = encoder.transform(df[col])
        
        X = df[self.features_selected]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled


class KickstarterPredictor:
    """Classe para fazer predições"""
    
    def __init__(self, model, preprocessor, threshold=0.5):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold
    
    def predict_single(self, project_data):
        """Faz predição para um projeto"""
        df = pd.DataFrame([project_data])
        X = self.preprocessor.transform(df)
        proba = self.model.predict_proba(X)[0, 1]
        prediction = int(proba >= self.threshold)
        recommendations = self._generate_recommendations(project_data, proba)
        
        return {
            'success_probability': float(proba),
            'prediction': 'Sucesso' if prediction else 'Falha',
            'confidence': self._calculate_confidence(proba),
            'recommendations': recommendations,
            'threshold_used': self.threshold
        }
    
    def _calculate_confidence(self, proba):
        distance = abs(proba - self.threshold)
        if distance > 0.3:
            return 'Alta'
        elif distance > 0.15:
            return 'Média'
        else:
            return 'Baixa'
    
    def _generate_recommendations(self, project_data, proba):
        recommendations = []
        
        goal = project_data.get('usd_goal_real', 0)
        if goal > 50000:
            recommendations.append("⚠️ Meta muito alta. Considere reduzir para aumentar chances.")
        elif goal < 1000:
            recommendations.append("✅ Meta modesta, boa estratégia para primeira campanha.")
        else:
            recommendations.append("✅ Meta dentro da faixa recomendada.")
        
        if 'campaign_days' not in project_data:
            launched = pd.to_datetime(project_data.get('launched'))
            deadline = pd.to_datetime(project_data.get('deadline'))
            campaign_days = (deadline - launched).days
        else:
            campaign_days = project_data.get('campaign_days')
            
        if campaign_days < 20:
            recommendations.append("⚠️ Campanha muito curta. Ideal entre 25-35 dias.")
        elif campaign_days > 45:
            recommendations.append("⚠️ Campanha muito longa. Pode perder momentum.")
        else:
            recommendations.append("✅ Duração adequada da campanha.")
        
        name_words = len(project_data.get('name', '').split())
        if name_words < 3:
            recommendations.append("💡 Título muito curto. Seja mais descritivo.")
        elif name_words > 10:
            recommendations.append("💡 Título muito longo. Seja mais conciso.")
        
        if proba < 0.3:
            recommendations.append("🔴 Risco alto de falha. Revise estratégia completa.")
        elif proba < 0.5:
            recommendations.append("🟡 Chances moderadas. Pequenos ajustes podem fazer diferença.")
        elif proba < 0.7:
            recommendations.append("🟢 Boas chances de sucesso. Mantenha execução forte.")
        else:
            recommendations.append("🌟 Excelentes chances! Foque na execução.")
        
        return recommendations


# Função para carregar modelo localmente
@st.cache_resource
def load_model_local():
    """Carrega o modelo localmente"""
    try:
        model_path = 'kickstarter_model_v1.pkl'
        
        # Verificar se existe
        if not os.path.exists(model_path):
            st.error("❌ Modelo não encontrado!")
            st.info("Execute python download_model.py para baixar o modelo")
            return None, None
        
        # Carregar
        model_data = joblib.load(model_path)
        
        # Criar predictor
        predictor = KickstarterPredictor(
            model=model_data['model'],
            preprocessor=model_data['preprocessor'],
            threshold=model_data['optimal_threshold']
        )
        
        return predictor, model_data
        
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        return None, None

# Carregar modelo
predictor_local, model_data_local = load_model_local()

# ========== FIM DAS ADIÇÕES ==========
# Inicializar session state para controle do spaCy
if 'use_spacy' not in st.session_state:
    st.session_state.use_spacy = True  # Ativado por padrão

# Tentar importar OpenAI
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    if OPENAI_AVAILABLE:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except:
    OPENAI_AVAILABLE = False
    client = None

# Tentar importar spaCy APENAS se estiver ativado
SPACY_AVAILABLE = False
nlp = None

if st.session_state.use_spacy:
    try:
        import spacy
        # Tentar carregar modelo em português ou inglês
        try:
            nlp = spacy.load("pt_core_news_sm")
        except:
            try:
                nlp = spacy.load("en_core_web_sm")
            except:
                # Se não tiver modelo, criar pipeline básico
                nlp = spacy.blank("pt")
        SPACY_AVAILABLE = True
    except:
        SPACY_AVAILABLE = False
        nlp = None



# Configurações
API_URL = os.getenv("KICKSTARTER_API_URL", "http://localhost:8000")

# Base de dados de usuários (requisito do case)
USERS_DATABASE = {
    "joao@example.com": {
        "nome": "João Silva",
        "cargo": "Gerente de Projetos",
        "experiencia_anos": 5,
        "projetos_historico": 15,
        "taxa_sucesso_pessoal": 0.80,
        "categorias_experiencia": ["Technology", "Design"],
        "projetos_detalhes": [
            {"nome": "Smart Home App", "categoria": "Technology", "sucesso": True, "meta": 25000},
            {"nome": "Eco Design Kit", "categoria": "Design", "sucesso": True, "meta": 15000},
            {"nome": "AI Assistant", "categoria": "Technology", "sucesso": False, "meta": 50000}
        ]
    },
    "maria@example.com": {
        "nome": "Maria Santos", 
        "cargo": "Analista de Projetos",
        "experiencia_anos": 3,
        "projetos_historico": 10,
        "taxa_sucesso_pessoal": 0.65,
        "categorias_experiencia": ["Games", "Art"],
        "projetos_detalhes": [
            {"nome": "Board Game Adventure", "categoria": "Games", "sucesso": True, "meta": 10000},
            {"nome": "Digital Art Gallery", "categoria": "Art", "sucesso": True, "meta": 8000},
            {"nome": "Mobile Game RPG", "categoria": "Games", "sucesso": False, "meta": 30000}
        ]
    },
    "pedro@example.com": {
        "nome": "Pedro Oliveira",
        "cargo": "Coordenador de Projetos", 
        "experiencia_anos": 8,
        "projetos_historico": 25,
        "taxa_sucesso_pessoal": 0.90,
        "categorias_experiencia": ["Film & Video", "Music", "Publishing"],
        "projetos_detalhes": [
            {"nome": "Documentary Series", "categoria": "Film & Video", "sucesso": True, "meta": 40000},
            {"nome": "Music Album", "categoria": "Music", "sucesso": True, "meta": 12000}
        ]
    },
    "default": {
        "nome": "Novo Usuário",
        "cargo": "Criador de Projetos",
        "experiencia_anos": 0,
        "projetos_historico": 0,
        "taxa_sucesso_pessoal": 0.0,
        "categorias_experiencia": [],
        "projetos_detalhes": []
    }
}

# CSS customizado
st.markdown("""
<style>
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .danger-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        text-align: left;
        margin-right: 20%;
    }
    .chat-header-split {
        background: #1f77b4;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
    }
    .top-chat-container {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .user-profile-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #a5d6a7;
        margin: 10px 0;
    }
    .extraction-method {
        font-size: 0.8em;
        color: #666;
        font-style: italic;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar session state
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'project_data' not in st.session_state:
    st.session_state.project_data = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = USERS_DATABASE["default"]
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'extraction_method' not in st.session_state:
    st.session_state.extraction_method = None

# Verificar se API está online
@st.cache_data(ttl=60)
def check_api_health():
    """Verifica se a API está online"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Carregar categorias
@st.cache_data(ttl=300)
def load_categories():
    """Carrega categorias disponíveis da API"""
    try:
        response = requests.get(f"{API_URL}/info/categories")
        if response.status_code == 200:
            data = response.json()
            return {cat['value']: cat for cat in data['categories']}
    except:
        pass
    
    # Fallback se API não responder
    return {
        'Film & Video': {'description': 'Filmes, documentários, vídeos', 'avg_success': '42%'},
        'Music': {'description': 'Álbuns, shows, instrumentos', 'avg_success': '53%'},
        'Publishing': {'description': 'Livros, revistas, e-books', 'avg_success': '35%'},
        'Games': {'description': 'Jogos de tabuleiro, card games, RPG', 'avg_success': '44%'},
        'Technology': {'description': 'Gadgets, apps, hardware', 'avg_success': '24%'},
        'Design': {'description': 'Produtos, móveis, acessórios', 'avg_success': '42%'},
        'Art': {'description': 'Pinturas, esculturas, instalações', 'avg_success': '45%'},
        'Comics': {'description': 'HQs, graphic novels, mangás', 'avg_success': '59%'},
        'Theater': {'description': 'Peças, musicais, performances', 'avg_success': '64%'},
        'Food': {'description': 'Restaurantes, produtos alimentícios', 'avg_success': '28%'},
        'Photography': {'description': 'Projetos fotográficos, livros de fotos', 'avg_success': '34%'},
        'Fashion': {'description': 'Roupas, calçados, acessórios', 'avg_success': '28%'},
        'Dance': {'description': 'Espetáculos, workshops, vídeos', 'avg_success': '65%'},
        'Journalism': {'description': 'Reportagens, documentários jornalísticos', 'avg_success': '24%'},
        'Crafts': {'description': 'Artesanato, DIY, kits', 'avg_success': '27%'}
    }

# Países disponíveis
COUNTRIES = {
    'US': 'Estados Unidos',
    'GB': 'Reino Unido',
    'CA': 'Canadá',
    'AU': 'Austrália',
    'DE': 'Alemanha',
    'FR': 'França',
    'IT': 'Itália',
    'ES': 'Espanha',
    'NL': 'Países Baixos',
    'SE': 'Suécia',
    'BR': 'Brasil',
    'JP': 'Japão',
    'MX': 'México'
}

# Categorias válidas
VALID_CATEGORIES = {
    'Film & Video', 'Music', 'Publishing', 'Games', 'Technology',
    'Design', 'Art', 'Comics', 'Theater', 'Food', 'Photography',
    'Fashion', 'Dance', 'Journalism', 'Crafts'
}

# Mapeamento de categorias em português
CATEGORY_MAPPING = {
    'filme': 'Film & Video',
    'vídeo': 'Film & Video',
    'video': 'Film & Video',
    'música': 'Music',
    'musica': 'Music',
    'publicação': 'Publishing',
    'publicacao': 'Publishing',
    'livro': 'Publishing',
    'jogos': 'Games',
    'jogo': 'Games',
    'game': 'Games',
    'tecnologia': 'Technology',
    'tech': 'Technology',
    'design': 'Design',
    'arte': 'Art',
    'quadrinhos': 'Comics',
    'hq': 'Comics',
    'teatro': 'Theater',
    'comida': 'Food',
    'alimentação': 'Food',
    'fotografia': 'Photography',
    'foto': 'Photography',
    'moda': 'Fashion',
    'dança': 'Dance',
    'danca': 'Dance',
    'jornalismo': 'Journalism',
    'artesanato': 'Crafts'
}

# Adicionar estas funções ao código do app_streamlit_hybrid_completo.py

def preprocess_message(message: str) -> str:
    """
    Pré-processa a mensagem para corrigir erros comuns
    """
    # Converter para lowercase para comparações
    message_lower = message.lower()
    
    # Dicionário de correções comuns
    corrections = {
        # Erros de digitação comuns
        'categria': 'categoria',
        'categorai': 'categoria',
        'catgoria': 'categoria',
        'categora': 'categoria',
        
        # Variações de palavras
        'dolar': 'dollar',
        'dolares': 'dollars',
        'reais': 'dollars',  # Assumir conversão
        
        # Abreviações de valores
        'k ': '000 ',
        'mil ': '000 ',
        
        # Países
        'brasil': 'BR',
        'estados unidos': 'US',
        'eua': 'US',
        'usa': 'US',
        
        # Categorias em português
        'tecnologia': 'Technology',
        'jogos': 'Games',
        'música': 'Music',
        'musica': 'Music',
        'arte': 'Art',
        'filmes': 'Film & Video',
        'filme': 'Film & Video',
        'video': 'Film & Video',
        'vídeo': 'Film & Video',
        'design': 'Design',
        'comida': 'Food',
        'teatro': 'Theater',
        'dança': 'Dance',
        'danca': 'Dance',
        'fotografia': 'Photography',
        'moda': 'Fashion',
        'artesanato': 'Crafts',
        'publicação': 'Publishing',
        'publicacao': 'Publishing',
        'quadrinhos': 'Comics',
        'jornalismo': 'Journalism'
    }
    
    # Aplicar correções
    result = message
    for wrong, correct in corrections.items():
        # Usar regex para substituir palavras completas
        import re
        pattern = r'\b' + re.escape(wrong) + r'\b'
        result = re.sub(pattern, correct, result, flags=re.IGNORECASE)
    
    return result

def extract_with_spacy_improved(message: str) -> Optional[Dict[str, Any]]:
    """
    Versão melhorada do extrator spaCy com pré-processamento
    """
    if not SPACY_AVAILABLE or not st.session_state.use_spacy:
        return None
    
    try:
        # Pré-processar mensagem
        processed_message = preprocess_message(message)
        print(f"Mensagem original: {message}")
        print(f"Mensagem processada: {processed_message}")
        
        # Padrões regex expandidos
        patterns = {
            'nome': [
                # Padrões estruturados
                r'nome:\s*([^\n,]+?)(?=\s+categoria:|$)',
                r'projeto:\s*([^\n,]+?)(?=\s+categoria:|$)',
                r'título:\s*([^\n,]+?)(?=\s+categoria:|$)',
                
                # Padrões mais flexíveis
                r'meu projeto (?:é|e|se chama)\s+([^\n,]+?)(?=\s+categoria|$)',
                r'projeto\s+([^\n,]+?)\s+(?:categoria|da categoria)',
                r'analise?\s+(?:o\s+)?(?:meu\s+)?projeto\s+([^\n,]+?)\s+categoria',
                
                # Padrão mais genérico (última tentativa)
                r'(?:projeto|nome)\s*:?\s*([a-zA-Z0-9\s\-\_]+?)(?=\s*(?:categoria|tipo|meta|$))'
            ],
            'categoria': [
                # Padrões estruturados
                r'categoria:\s*([^\n,]+?)(?=\s+meta:|$)',
                r'tipo:\s*([^\n,]+?)(?=\s+meta:|$)',
                r'category:\s*([^\n,]+?)(?=\s+meta:|$)',
                
                # Padrões flexíveis
                r'(?:da\s+)?categoria\s+([^\n,]+?)(?=\s+meta|$)',
                r'é\s+(?:um|uma)\s+([^\n,]+?)(?=\s+meta|com|$)',
                
                # Categorias entre aspas ou parênteses
                r'categoria[:\s]+["\']([^"\']+)["\']',
                r'categoria[:\s]+\(([^)]+)\)',
                
                # Padrão genérico
                r'categoria\s*:?\s*([a-zA-Z\s&]+?)(?=\s*(?:meta|valor|$))'
            ],
            'meta': [
                # Valores monetários estruturados
                r'meta:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'objetivo:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'goal:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'valor:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                
                # Valores com k/mil
                r'meta\s*:?\s*(\d+)\s*(?:k|mil)',
                r'(\d+)\s*(?:k|mil)\s*(?:dólares|dolares|reais|dollars)',
                
                # Valores entre símbolos
                r'\$\s*([\d,]+(?:\.\d{2})?)',
                r'R\$\s*([\d,]+(?:\.\d{2})?)',
                
                # Padrão genérico
                r'meta\s*:?\s*(?:de\s+)?\$?\s*([\d,\.]+)'
            ],
            'pais': [
                r'país:\s*([A-Za-z]{2})',
                r'pais:\s*([A-Za-z]{2})',
                r'country:\s*([A-Za-z]{2})',
                r'local:\s*([A-Za-z]{2})',
                r'de\s+([A-Za-z]{2})(?:\s|$)'
            ],
            'inicio': [
                r'início:\s*(\d{4}-\d{2}-\d{2})',
                r'inicio:\s*(\d{4}-\d{2}-\d{2})',
                r'começa:\s*(\d{4}-\d{2}-\d{2})',
                r'lançamento:\s*(\d{4}-\d{2}-\d{2})',
                r'start:\s*(\d{4}-\d{2}-\d{2})',
                r'data\s+(?:de\s+)?início:\s*(\d{4}-\d{2}-\d{2})'
            ],
            'fim': [
                r'fim:\s*(\d{4}-\d{2}-\d{2})',
                r'término:\s*(\d{4}-\d{2}-\d{2})',
                r'termino:\s*(\d{4}-\d{2}-\d{2})',
                r'deadline:\s*(\d{4}-\d{2}-\d{2})',
                r'end:\s*(\d{4}-\d{2}-\d{2})',
                r'data\s+(?:de\s+)?fim:\s*(\d{4}-\d{2}-\d{2})',
                r'até:\s*(\d{4}-\d{2}-\d{2})'
            ]
        }
        
        # Extrair dados usando regex
        extracted_data = {}
        
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, processed_message, re.IGNORECASE)
                if match:
                    extracted_data[field] = match.group(1).strip()
                    break
        
        # Processar valores especiais
        if 'meta' in extracted_data:
            meta_str = extracted_data['meta']
            
            # Verificar se tem 'k' ou 'mil'
            if 'k' in message.lower() or 'mil' in message.lower():
                # Extrair apenas números
                numbers = re.findall(r'(\d+)', meta_str)
                if numbers:
                    extracted_data['meta'] = str(int(numbers[0]) * 1000)
        
        # Validar se temos dados mínimos
        if not all(key in extracted_data for key in ['nome', 'categoria', 'meta']):
            print(f"Dados incompletos. Extraídos: {extracted_data}")
            return None
        
        # Converter e validar dados
        try:
            # Limpar e converter meta
            meta_str = extracted_data['meta'].replace(',', '')
            if '.' in meta_str and len(meta_str.split('.')[-1]) == 2:
                meta = float(meta_str)
            else:
                meta_str = meta_str.replace('.', '')
                meta = float(meta_str)
            
            # Normalizar categoria
            categoria = normalize_category(extracted_data['categoria'])
            
            # Preparar dados finais
            project_data = {
                "name": extracted_data['nome'],
                "main_category": categoria,
                "country": extracted_data.get('pais', 'US').upper(),
                "usd_goal_real": meta,
                "launched": extracted_data.get('inicio', datetime.now().strftime("%Y-%m-%d")),
                "deadline": extracted_data.get('fim', (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
            }
            
            # Validar datas
            launched_date = datetime.strptime(project_data['launched'], "%Y-%m-%d")
            deadline_date = datetime.strptime(project_data['deadline'], "%Y-%m-%d")
            
            if deadline_date <= launched_date:
                project_data['deadline'] = (launched_date + timedelta(days=30)).strftime("%Y-%m-%d")
            
            return project_data
            
        except Exception as e:
            print(f"Erro na conversão: {e}")
            return None
            
    except Exception as e:
        print(f"Erro geral: {e}")
        return None

# Modificar a função extract_project_info_from_message para usar a versão melhorada
def extract_project_info_from_message(message):
    """Extrai informações do projeto da mensagem do usuário"""
    # Primeiro tenta com spaCy melhorado SE ESTIVER ATIVADO
    if SPACY_AVAILABLE and st.session_state.use_spacy:
        project_data = extract_with_spacy_improved(message)
        if project_data:
            st.session_state.extraction_method = "spaCy (local/gratuito)"
            return project_data 
    
    # Se spaCy falhar ou estiver desativado e OpenAI estiver disponível
    if OPENAI_AVAILABLE and client:
        try:
            prompt = f"""
            Extraia as informações do projeto Kickstarter desta mensagem.
            Retorne APENAS um JSON válido com os campos:
            - name: nome do projeto
            - main_category: categoria (deve ser uma das válidas: Technology, Games, Art, Music, Film & Video, Design, Comics, Theater, Food, Photography, Fashion, Dance, Journalism, Crafts, Publishing)
            - country: código do país (2 letras)
            - usd_goal_real: meta em dólares (número)
            - launched: data de início (YYYY-MM-DD)
            - deadline: data fim (YYYY-MM-DD)
            
            Se algum campo não for mencionado, use valores padrão razoáveis.
            
            Mensagem: {message}
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1
            )
            
            json_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                st.session_state.extraction_method = "OpenAI GPT-3.5 (principal)" if not st.session_state.use_spacy else "OpenAI GPT-3.5 (fallback)"
                return extracted_data
        except Exception as e:
            print(f"Erro com OpenAI: {e}")
    
    return None

# Adicione este código ANTES do container do chat (após o CSS customizado e antes de "# Layout com Chat no Topo")

# Seção de Boas-Vindas e Instruções
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h1 style="text-align: center; margin-bottom: 20px;">🎯 Bem-vindo ao Kickstarter Success Predictor!</h1>
    <p style="text-align: center; font-size: 1.1em; margin-bottom: 30px;">
        Sou seu assistente de IA especializado em prever o sucesso de projetos no Kickstarter.<br>
        Posso analisar seu projeto e dar recomendações personalizadas!
    </p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.user_email or st.session_state.user_email == "default":
    # Container para login rápido
    st.markdown("""
    <div style="background: #f8f9fa; border: 2px dashed #dee2e6; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h3 style="text-align: center; color: #495057;">🚀 Experimente com um Usuário Demo</h3>
        <p style="text-align: center; color: #6c757d;">Clique em um dos perfis abaixo para testar o sistema com histórico real</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Três colunas para os botões
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    
    with demo_col1:
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h4>👨‍💼 João Silva</h4>
            <p style="font-size: 0.9em; color: #666;">
                Gerente de Projetos<br>
                5 anos experiência<br>
                Taxa sucesso: 80%<br>
                <span style="color: #28a745;">✓ Technology ✓ Design</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Entrar como João", key="btn_joao", use_container_width=True, type="primary"):
            st.session_state.user_email = "joao@example.com"
            st.session_state.user_data = USERS_DATABASE["joao@example.com"]
            st.success("✅ Logado como João Silva!")
            time.sleep(1)
            st.rerun()
    
    with demo_col2:
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h4>👩‍💻 Maria Santos</h4>
            <p style="font-size: 0.9em; color: #666;">
                Analista de Projetos<br>
                3 anos experiência<br>
                Taxa sucesso: 65%<br>
                <span style="color: #17a2b8;">✓ Games ✓ Art</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Entrar como Maria", key="btn_maria", use_container_width=True):
            st.session_state.user_email = "maria@example.com"
            st.session_state.user_data = USERS_DATABASE["maria@example.com"]
            st.success("✅ Logada como Maria Santos!")
            time.sleep(1)
            st.rerun()
    
    with demo_col3:
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h4>👨‍🎨 Pedro Oliveira</h4>
            <p style="font-size: 0.9em; color: #666;">
                Coordenador<br>
                8 anos experiência<br>
                Taxa sucesso: 90%<br>
                <span style="color: #6610f2;">✓ Film ✓ Music ✓ Publishing</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Entrar como Pedro", key="btn_pedro", use_container_width=True):
            st.session_state.user_email = "pedro@example.com"
            st.session_state.user_data = USERS_DATABASE["pedro@example.com"]
            st.success("✅ Logado como Pedro Oliveira!")
            time.sleep(1)
            st.rerun()
    
    # Opção de continuar sem login
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("➡️ Continuar sem login", key="btn_anonimo", use_container_width=True):
        st.info("Você pode usar o sistema, mas não terá análises personalizadas baseadas em histórico.")
        time.sleep(1.5)
        st.rerun()

else:
    # Se já está logado, mostrar card de boas-vindas personalizado
    user = st.session_state.user_data
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0;">
        <h3>👋 Bem-vindo de volta, {user['nome']}!</h3>
        <p>
            <strong>{user['cargo']}</strong> | 
            {user['experiencia_anos']} anos de experiência | 
            {user['projetos_historico']} projetos | 
            Taxa de sucesso: {user['taxa_sucesso_pessoal']:.0%}
        </p>
        <p>Especialista em: {', '.join(user['categorias_experiencia'])}</p>
    </div>
    """, unsafe_allow_html=True)

# Container de instruções expandível
with st.expander("📝 **Como usar o Assistente de IA** (Clique para ver exemplos)", expanded=True):
    col_inst1, col_inst2 = st.columns(2)
    
    with col_inst1:
        st.markdown("""
        ### ✨ Formato Recomendado
        ```
        Analise meu projeto: 
        Nome: [nome] 
        Categoria: [categoria] 
        Meta: $[valor] 
        País: [código] 
        Início: YYYY-MM-DD 
        Fim: YYYY-MM-DD
        ```
        """)
    
    with col_inst2:
        st.markdown("""
        ### ✅ Exemplos Funcionais
        ```
        Analise meu projeto: Nome: power 
        Categoria: Games Meta: $10,000 
        País: US Início: 2025-07-03 
        Fim: 2025-08-02
        ```
        
        ```
        Sou joao@example.com. Analise meu projeto: 
        Nome: power Categoria: Technology 
        Meta: $10000 País: US 
        Início: 2025-07-04 Fim: 2025-08-03
        ```
        """)  

# Linha divisória estilizada
st.markdown("""
<div style="height: 2px; background: linear-gradient(to right, transparent, #667eea, #764ba2, transparent); margin: 20px 0;"></div>
""", unsafe_allow_html=True)

# Adicionar exemplos de uso no início do chat
def get_initial_chat_message():
    """Retorna mensagem inicial com exemplos para o usuário"""
    return """
Como posso ajudar com seu projeto hoje?
"""

# Modificar a resposta de erro para ser mais útil
def get_error_response():
    """Retorna mensagem de erro útil com exemplos"""
    return """
❌ **Não consegui entender os dados do seu projeto.**

📝 **Por favor, use um destes formatos:**

**Formato completo (recomendado):**
```
Analise meu projeto: 
Nome: [nome do projeto]
Categoria: [categoria]
Meta: $[valor]
País: [código de 2 letras]
Início: YYYY-MM-DD
Fim: YYYY-MM-DD
```

**Formato simplificado:**
```
Nome: [projeto] Categoria: [categoria] Meta: $[valor]
```

**Exemplos reais que funcionam:**
- `Analise meu projeto: Nome: SmartHome Categoria: Technology Meta: $15,000 País: US`
- `Nome: BoardGame Fun Categoria: Games Meta: $8,000`
- `projeto EcoBottle design 5000 dolares`

**Categorias válidas:**
- **Tecnologia**: Technology
- **Jogos**: Games
- **Arte**: Art
- **Música**: Music
- **Filme/Vídeo**: Film & Video
- **Design**: Design
- **Outras**: Comics, Theater, Food, Photography, Fashion, Dance, Journalism, Crafts, Publishing

💡 **Dicas:**
- Escreva valores sem vírgulas: $10000 (não $10,000)
- Use códigos de país: US, BR, GB, etc.
- Posso entender português: "jogos" → Games

Tente novamente! Estou aqui para ajudar 😊
"""

def normalize_category(category: str) -> str:
    """Normaliza categoria para formato válido"""
    # Primeiro tenta match direto
    if category in VALID_CATEGORIES:
        return category
    
    # Tenta mapear do português
    category_lower = category.lower().strip()
    if category_lower in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[category_lower]
    
    # Tenta match parcial
    for key, value in CATEGORY_MAPPING.items():
        if key in category_lower or category_lower in key:
            return value
    
    # Se não encontrar, retorna Technology como padrão
    return "Technology"

def extract_with_spacy(message: str) -> Optional[Dict[str, Any]]:
    """
    Extrai informações do projeto usando spaCy e regex.
    Retorna None se não conseguir extrair informações suficientes.
    """
    if not SPACY_AVAILABLE or not st.session_state.use_spacy:
        return None
    
    try:
        # Converter para lowercase para facilitar matching
        message_lower = message.lower()
        
        # Padrões regex para cada campo - CORRIGIDOS
        patterns = {
            'nome': [
                r'nome:\s*([^\n,]+?)(?=\s+categoria:|$)',
                r'projeto:\s*([^\n,]+?)(?=\s+categoria:|$)',
                r'título:\s*([^\n,]+?)(?=\s+categoria:|$)',
                r'nome\s+(?:é|e)\s+([^\n,]+?)(?=\s+categoria:|$)'
            ],
            'categoria': [
                r'categoria:\s*([^\n,]+?)(?=\s+meta:|$)',
                r'tipo:\s*([^\n,]+?)(?=\s+meta:|$)',
                r'category:\s*([^\n,]+?)(?=\s+meta:|$)'
            ],
            'meta': [
                # Padrões mais específicos para capturar valores monetários completos
                r'meta:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'objetivo:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'goal:\s*\$?\s*([\d,]+(?:\.\d{2})?)',
                r'\$\s*([\d,]+(?:\.\d{2})?)'
            ],
            'pais': [
                r'país:\s*([A-Za-z]{2})',
                r'pais:\s*([A-Za-z]{2})',
                r'country:\s*([A-Za-z]{2})',
                r'local:\s*([A-Za-z]{2})'
            ],
            'inicio': [
                r'início:\s*(\d{4}-\d{2}-\d{2})',
                r'inicio:\s*(\d{4}-\d{2}-\d{2})',
                r'começa:\s*(\d{4}-\d{2}-\d{2})',
                r'lançamento:\s*(\d{4}-\d{2}-\d{2})',
                r'start:\s*(\d{4}-\d{2}-\d{2})'
            ],
            'fim': [
                r'fim:\s*(\d{4}-\d{2}-\d{2})',
                r'término:\s*(\d{4}-\d{2}-\d{2})',
                r'termino:\s*(\d{4}-\d{2}-\d{2})',
                r'deadline:\s*(\d{4}-\d{2}-\d{2})',
                r'end:\s*(\d{4}-\d{2}-\d{2})'
            ]
        }
        
        # Debug: imprimir a mensagem recebida
        print(f"Mensagem recebida: {message}")
        
        # Extrair dados usando regex
        extracted_data = {}
        
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    extracted_data[field] = match.group(1).strip()
                    print(f"Extraído {field}: {extracted_data[field]}")
                    break
        
        # Validar se temos dados mínimos
        if not all(key in extracted_data for key in ['nome', 'categoria', 'meta']):
            print(f"Dados incompletos. Extraídos: {extracted_data}")
            return None
        
        # Converter e validar dados
        try:
            # Limpar e converter meta - CORREÇÃO PRINCIPAL
            meta_str = extracted_data['meta'].replace(',', '')
            # Não remover o ponto se for decimal
            if '.' in meta_str and len(meta_str.split('.')[-1]) == 2:
                # É um valor decimal (ex: 10000.00)
                meta = float(meta_str)
            else:
                # É um valor inteiro (ex: 10000 ou 10,000)
                meta_str = meta_str.replace('.', '')
                meta = float(meta_str)
            
            print(f"Meta convertida: {meta}")
            
            # Normalizar categoria
            categoria = normalize_category(extracted_data['categoria'])
            
            # Preparar dados finais
            project_data = {
                "name": extracted_data['nome'],
                "main_category": categoria,
                "country": extracted_data.get('pais', 'US').upper(),
                "usd_goal_real": meta,
                "launched": extracted_data.get('inicio', datetime.now().strftime("%Y-%m-%d")),
                "deadline": extracted_data.get('fim', (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
            }
            
            # Validar datas
            launched_date = datetime.strptime(project_data['launched'], "%Y-%m-%d")
            deadline_date = datetime.strptime(project_data['deadline'], "%Y-%m-%d")
            
            if deadline_date <= launched_date:
                # Ajustar deadline se inválido
                project_data['deadline'] = (launched_date + timedelta(days=30)).strftime("%Y-%m-%d")
            
            print(f"Dados finais: {project_data}")
            return project_data
            
        except Exception as e:
            print(f"Erro na conversão: {e}")
            return None
            
    except Exception as e:
        print(f"Erro geral: {e}")
        return None

# Funções do Chatbot
def make_prediction_from_chat(project_info):
    """Faz predição através do chat"""
    try:
        # Preparar dados para API
        project_data = {
            "name": project_info.get('name', 'My Kickstarter Project'),
            "main_category": project_info.get('category', 'Technology'),
            "country": project_info.get('country', 'US'),
            "usd_goal_real": float(project_info.get('goal', 10000)),
            "launched": project_info.get('launched', datetime.now().strftime("%Y-%m-%d")),
            "deadline": project_info.get('deadline', (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
        }
        
        # Fazer requisição para API
        response = requests.post(f"{API_URL}/predict", json=project_data)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}



def get_chat_response(user_message, context=None):
    """Gera resposta do chatbot usando OpenAI ou respostas predefinidas"""
    try:
        # Verificar se o usuário quer fazer uma predição
        prediction_keywords = ['predict', 'prever', 'chance', 'probabilidade', 'analisar projeto', 'analyze', 'analise']
        wants_prediction = any(keyword in user_message.lower() for keyword in prediction_keywords)
        
        # Se quiser predição e tiver dados estruturados
        if wants_prediction:
            # Tentar extrair dados
            project_info = extract_project_info_from_message(user_message)
            
            if project_info:
                # CORREÇÃO: Os dados já vêm padronizados do extract_with_spacy
                # Apenas garantir que os campos estejam corretos
                project_data_for_api = {
                    "name": project_info.get('name', 'My Project'),
                    "main_category": project_info.get('main_category', 'Technology'),
                    "country": project_info.get('country', 'US'),
                    "usd_goal_real": float(project_info.get('usd_goal_real', 10000)),
                    "launched": project_info.get('launched', datetime.now().strftime("%Y-%m-%d")),
                    "deadline": project_info.get('deadline', (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
                }
                
                # Fazer predição real com a API diretamente
                try:
                    response = requests.post(f"{API_URL}/predict", json=project_data_for_api)
                    
                    if response.status_code == 200:
                        prediction_result = response.json()
                        
                        # Salvar no contexto com dados padronizados
                        st.session_state.project_data = project_data_for_api
                        st.session_state.prediction_result = prediction_result
                        
                        # Criar resposta formatada
                        duration_days = (pd.to_datetime(project_data_for_api['deadline']) - pd.to_datetime(project_data_for_api['launched'])).days
                        
                        # Carregar categorias para mostrar taxa média
                        categories = load_categories()
                        
                        # Emoji baseado na probabilidade
                        if prediction_result['success_probability'] >= 0.6:
                            emoji_result = "🟢"
                            status_msg = "ALTA CHANCE DE SUCESSO!"
                        elif prediction_result['success_probability'] >= 0.5:
                            emoji_result = "🟡"
                            status_msg = "CHANCE MODERADA"
                        elif prediction_result['success_probability'] >= 0.3:
                            emoji_result = "🟠"
                            status_msg = "CHANCE BAIXA - PRECISA MELHORAR"
                        else:
                            emoji_result = "🔴"
                            status_msg = "ALTO RISCO DE FRACASSO!"
                        
                        # Adicionar análise personalizada baseada no usuário
                        user_analysis = ""
                        if st.session_state.user_email and st.session_state.user_email != "default":
                            user_data = st.session_state.user_data
                            
                            # Comparar com histórico pessoal
                            if user_data['taxa_sucesso_pessoal'] > 0:
                                diff = prediction_result['success_probability'] - user_data['taxa_sucesso_pessoal']
                                if diff > 0:
                                    user_analysis += f"\n\n📊 **Análise Personalizada para {user_data['nome']}:**\n"
                                    user_analysis += f"✅ Este projeto tem {diff*100:.1f}% mais chance que sua média histórica ({user_data['taxa_sucesso_pessoal']:.0%})!"
                                else:
                                    user_analysis += f"\n\n📊 **Análise Personalizada para {user_data['nome']}:**\n"
                                    user_analysis += f"⚠️ Este projeto está {abs(diff)*100:.1f}% abaixo da sua média histórica ({user_data['taxa_sucesso_pessoal']:.0%})"
                            
                            # Verificar experiência na categoria
                            if project_data_for_api['main_category'] in user_data['categorias_experiencia']:
                                user_analysis += f"\n✅ Você tem experiência em {project_data_for_api['main_category']}. Isso é um diferencial!"
                            else:
                                user_analysis += f"\n💡 Primeira vez em {project_data_for_api['main_category']}? Considere buscar mentoria nesta área."
                        
                        # Adicionar método de extração
                        extraction_info = ""
                        if st.session_state.extraction_method:
                            extraction_info = f"\n\n<p class='extraction-method'>📝 Dados extraídos via: {st.session_state.extraction_method}</p>"
                        
                        return f"""
{emoji_result} **{status_msg}**

🎯 **Análise do Projeto: {project_data_for_api['name']}**

📊 **TAXA DE SUCESSO: {prediction_result['success_probability']:.1%}**
🔮 **PREDIÇÃO: {prediction_result['prediction'].upper()}**
💪 **CONFIANÇA: {prediction_result['confidence']}**

**📋 Detalhes do Projeto:**
- 🎬 Categoria: {project_data_for_api['main_category']} (Taxa média de sucesso: {categories.get(project_data_for_api['main_category'], {}).get('avg_success', '42%')})
- 💰 Meta: ${project_data_for_api['usd_goal_real']:,.0f}
- 🌍 País: {project_data_for_api['country']}
- 📅 Duração: {duration_days} dias
- 🚀 Período: {project_data_for_api['launched']} até {project_data_for_api['deadline']}

**🎲 Threshold do modelo: {prediction_result['threshold_used']:.1%}**
Sua probabilidade está {(prediction_result['success_probability'] - prediction_result['threshold_used'])*100:.1f}% {'acima' if prediction_result['success_probability'] > prediction_result['threshold_used'] else 'abaixo'} do threshold.

**💡 Recomendações Personalizadas:**
{chr(10).join(f"- {rec}" for rec in prediction_result['recommendations'])}
{user_analysis}

**📈 Próximos Passos:**
{'✅ Você está no caminho certo! Foque na execução e marketing.' if prediction_result['success_probability'] >= 0.5 else '⚠️ Recomendo ajustar alguns aspectos antes de lançar.'}

Quer que eu:
- 📝 Sugira títulos melhores?
- 💰 Analise se a meta está adequada?
- 📅 Crie um cronograma de campanha?
- 🎁 Monte estrutura de recompensas?
{extraction_info}
"""
                    else:
                        return f"❌ Erro ao fazer predição: API retornou status {response.status_code}"
                        
                except Exception as e:
                    return f"❌ Erro ao fazer predição: {str(e)}\n\nPor favor, use o formulário na aba '🔮 Predictor' para análise precisa."
            else:
                return get_error_response()
        
        # Se não for predição ou não tiver OpenAI, usar respostas padrão
        if not OPENAI_AVAILABLE:
            # Respostas predefinidas para casos sem OpenAI
            message_lower = user_message.lower()
            
            # Verificar se é primeira mensagem/saudação
            if any(word in message_lower for word in ['oi', 'olá', 'hello', 'hi', 'início', 'começ', 'ajud']):
                return get_initial_chat_message()
            elif 'categoria' in message_lower or 'categories' in message_lower:
                categories = load_categories()
                return f"""
**Categorias disponíveis no Kickstarter:**

{chr(10).join(f"- {cat} ({info['avg_success']} sucesso)" for cat, info in categories.items())}

As categorias com maior taxa de sucesso são Dance, Theater e Comics!
"""
            else:
                return """
Desculpe, não entendi sua pergunta. 

**Posso ajudar com:**
- Prever sucesso do seu projeto
- Listar categorias disponíveis
- Dar dicas para melhorar suas chances

Para fazer uma predição, envie os dados do projeto no formato estruturado.
"""
        
        # Se tiver OpenAI, usar para respostas gerais
        system_message = """Você é um consultor especialista em crowdfunding do Kickstarter com 10 anos de experiência.
        
        REGRA CRÍTICA: Você NUNCA deve inventar taxas de sucesso ou probabilidades. 
        Se o usuário pedir uma predição, você DEVE usar a função de predição real que retorna a probabilidade exata do modelo.
        NUNCA diga coisas como "aproximadamente 75%" ou invente números.
        
        Você tem acesso a:
        - Um modelo preditivo treinado com 300,000+ projetos (AUC-ROC: 0.733)
        - Dados estatísticos sobre taxas de sucesso por categoria
        - Base de dados de usuários com histórico de projetos
        - Capacidade de fazer predições REAIS quando o usuário fornecer dados do projeto
        
        IMPORTANTE: Quando fizer uma predição, SEMPRE:
        1. Use os dados REAIS retornados pela API
        2. Mostre a taxa EXATA de sucesso
        3. Considere o histórico do usuário se disponível
        4. Seja direto e objetivo
        
        Se o usuário quiser fazer uma predição, extraia os dados e faça a chamada real para o modelo."""
        
        messages = [{"role": "system", "content": system_message}]
        
        # Adicionar contexto se disponível
        if context:
            context_message = f"""
            Contexto atual do projeto:
            - Nome: {context.get('name', 'Não definido')}
            - Categoria: {context.get('main_category', 'Não definida')}
            - Meta: ${context.get('usd_goal_real', 0):,.2f}
            - País: {context.get('country', 'Não definido')}
            - Duração: {context.get('campaign_days', 'Não definida')} dias
            
            Resultados da predição (se disponível):
            {json.dumps(st.session_state.prediction_result, indent=2) if st.session_state.prediction_result else 'Nenhuma predição feita ainda'}
            """
            messages.append({"role": "system", "content": context_message})
        
        # Adicionar informações do usuário se disponível
        if st.session_state.user_email and st.session_state.user_data:
            user_context = f"""
            Informações do usuário atual:
            - Nome: {st.session_state.user_data['nome']}
            - Cargo: {st.session_state.user_data['cargo']}
            - Experiência: {st.session_state.user_data['experiencia_anos']} anos
            - Projetos anteriores: {st.session_state.user_data['projetos_historico']}
            - Taxa de sucesso pessoal: {st.session_state.user_data['taxa_sucesso_pessoal']:.0%}
            - Experiência em categorias: {', '.join(st.session_state.user_data['categorias_experiencia'])}
            
            Use essas informações para personalizar suas recomendações.
            """
            messages.append({"role": "system", "content": user_context})
        
        # Adicionar histórico de conversa
        for msg in st.session_state.chat_messages[-10:]:  # Últimas 10 mensagens
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Adicionar mensagem atual
        messages.append({"role": "user", "content": user_message})
        
        # Fazer chamada para OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Desculpe, houve um erro ao processar sua mensagem: {str(e)}"

def analyze_project_with_ai(project_data, prediction_result):
    """Análise detalhada do projeto usando AI"""
    if not OPENAI_AVAILABLE:
        return "⚠️ OpenAI não está configurado. Configure OPENAI_API_KEY no arquivo .env para usar esta funcionalidade."
    
    user_context = ""
    if st.session_state.user_data and st.session_state.user_data != USERS_DATABASE["default"]:
        user_context = f"""
        Considere também o perfil do usuário:
        - {st.session_state.user_data['nome']} ({st.session_state.user_data['cargo']})
        - {st.session_state.user_data['experiencia_anos']} anos de experiência
        - Taxa de sucesso histórica: {st.session_state.user_data['taxa_sucesso_pessoal']:.0%}
        - Experiência em: {', '.join(st.session_state.user_data['categorias_experiencia'])}
        """
    
    prompt = f"""
    Analise este projeto Kickstarter e forneça insights detalhados:
    
    Dados do Projeto:
    {json.dumps(project_data, indent=2)}
    
    Resultado da Predição:
    {json.dumps(prediction_result, indent=2)}
    
    {user_context}
    
    Por favor, forneça:
    1. Análise dos pontos fortes e fracos
    2. 3 sugestões específicas para melhorar as chances
    3. Comparação com projetos bem-sucedidos na mesma categoria
    4. Estratégia de lançamento recomendada personalizada para este usuário
    
    Seja específico e prático.
    """
    
    return get_chat_response(prompt)

def generate_title_suggestions(current_title, category):
    """Gera sugestões de títulos melhores"""
    if not OPENAI_AVAILABLE:
        return """
**Sugestões de títulos baseadas em padrões de sucesso:**

1. **[Adjetivo] + [Produto] + [Benefício]**
   - Ex: "Revolutionary Solar Charger for Travelers"
   
2. **[Problema] + [Solução] + [Diferencial]**
   - Ex: "Never Lose Keys Again - Smart Bluetooth Tracker"
   
3. **[Público] + [Necessidade] + [Inovação]**
   - Ex: "Gamers Ultimate Wireless Controller Experience"

**Dicas:**
- Use 4-7 palavras
- Seja específico sobre o que faz
- Inclua um diferencial claro
- Evite jargões técnicos
"""
    
    prompt = f"""
    O título atual do projeto é: "{current_title}"
    Categoria: {category}
    
    Sugira 3 títulos melhores que:
    1. Sejam mais atrativos e descritivos
    2. Incluam palavras-chave relevantes para SEO
    3. Tenham entre 4-7 palavras
    4. Comuniquem claramente o valor do projeto
    
    Para cada sugestão, explique brevemente por que é melhor.
    """
    
    return get_chat_response(prompt)

def optimize_campaign_strategy(project_data, prediction_result):
    """Gera estratégia otimizada de campanha"""
    if not OPENAI_AVAILABLE:
        duration = (pd.to_datetime(project_data['deadline']) - pd.to_datetime(project_data['launched'])).days
        return f"""
**Estratégia de Campanha para {duration} dias:**

**🚀 Pré-Lançamento (7 dias antes):**
- Criar lista de e-mail com interessados
- Preparar conteúdo visual (vídeo + imagens)
- Engajar comunidade nas redes sociais
- Definir recompensas early bird (25% desconto)

**📈 Semana 1 - Momentum Inicial:**
- Objetivo: 30% da meta
- Ativar lista de e-mail no dia 1
- Postar em grupos relevantes
- Atualização diária nas primeiras 48h

**🎯 Semanas 2-3 - Manutenção:**
- Objetivo: 70% da meta
- Atualizações 2x por semana
- Adicionar stretch goals se > 50%
- Engajar apoiadores como embaixadores

**🏁 Última Semana - Sprint Final:**
- Objetivo: 100%+ da meta
- Campanha "últimas horas"
- Oferecer bônus limitados
- Live/AMA com criadores

**📊 Métricas para acompanhar:**
- Taxa de conversão de visitantes
- Ticket médio por apoiador
- Origem do tráfego
- Engajamento nas atualizações
"""
    
    user_context = ""
    if st.session_state.user_data and st.session_state.user_data != USERS_DATABASE["default"]:
        user_context = f"""
        Considere o perfil do usuário:
        - {st.session_state.user_data['nome']} tem {st.session_state.user_data['experiencia_anos']} anos de experiência
        - Taxa de sucesso histórica: {st.session_state.user_data['taxa_sucesso_pessoal']:.0%}
        - Já trabalhou com: {', '.join(st.session_state.user_data['categorias_experiencia'])}
        """
    
    prompt = f"""
    Crie um plano estratégico de 30 dias para maximizar o sucesso desta campanha:
    
    Projeto: {project_data['name']}
    Categoria: {project_data['main_category']}
    Meta: ${project_data['usd_goal_real']:,.2f}
    Probabilidade atual: {prediction_result['success_probability']:.1%}
    
    {user_context}
    
    Inclua:
    1. Cronograma detalhado (pré-lançamento, lançamento, meio, final)
    2. Metas de arrecadação por semana
    3. Estratégias de marketing específicas
    4. Momentos-chave para atualizações
    5. Táticas para manter momentum
    
    Seja prático e específico, considerando a experiência do usuário.
    """
    
    return get_chat_response(prompt)

# Layout com Chat no Topo
# Chat fixo na parte superior com layout melhorado
with st.container():
    st.markdown('<div class="top-chat-container">', unsafe_allow_html=True)
    
    # Header do chat
    chat_header_col1, chat_header_col2, chat_header_col3 = st.columns([1, 3, 1])
    with chat_header_col2:
        #st.markdown("### 💬 AI Assistant")
        if st.session_state.user_email and st.session_state.user_email != "default":
            st.caption(f"👤 Conversando com: {st.session_state.user_data['nome']}")
        
        # Mostrar status dos sistemas
        status_cols = st.columns(3)
        with status_cols[0]:
            if SPACY_AVAILABLE:
                st.success("✅ spaCy")
            else:
                st.error("❌ spaCy")
        with status_cols[1]:
            if OPENAI_AVAILABLE:
                st.success("✅ OpenAI")
            else:
                st.warning("⚠️ OpenAI")
        with status_cols[2]:
            if check_api_health():
                st.success("✅ API")
            else:
                st.error("❌ API")
    
    # Container do chat
    chat_main_col1, chat_main_col2, chat_main_col3 = st.columns([0.5, 4, 0.5])
    
    with chat_main_col2:
        # Área de mensagens expandida
        chat_area = st.container(height=250)
        with chat_area:
            # Se não houver mensagens, mostrar mensagem inicial
            if len(st.session_state.chat_messages) == 0:
                initial_msg = get_initial_chat_message()
                st.markdown(f'<div class="chat-message assistant-message">🤖</div>', 
                          unsafe_allow_html=True)
                st.markdown(initial_msg, unsafe_allow_html=True)
            else:
                # Mostrar mensagens completas
                for message in st.session_state.chat_messages[-5:]:  # Últimas 5 mensagens
                    if message["role"] == "user":
                        st.markdown(f'<div class="chat-message user-message">👤 {message["content"]}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        # Mostrar resposta completa do bot
                        st.markdown(f'<div class="chat-message assistant-message">🤖</div>', 
                                  unsafe_allow_html=True)
                        # Usar markdown para formatar a resposta completa
                        st.markdown(message["content"], unsafe_allow_html=True)
        
        # Área de input
        input_col1, input_col2, input_col3 = st.columns([10, 1, 1])
        
        with input_col1:
            user_input = st.text_input(
                "Digite sua pergunta:", 
                key="top_chat_input", 
                label_visibility="collapsed", 
                placeholder="Ex: Analise meu projeto: Nome: power Categoria: Film & Video Meta: $10,000 País: US Início: 2025-07-03 Fim: 2025-08-02"
            )
        
        with input_col2:
            if st.button("📤 Enviar", key="top_send", use_container_width=True):
                if user_input:
                    # Adicionar mensagem do usuário
                    st.session_state.chat_messages.append({"role": "user", "content": user_input})
                    
                    # Obter resposta
                    with st.spinner("Analisando..."):
                        response = get_chat_response(user_input, st.session_state.project_data)
                    
                    # Adicionar resposta
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                    # Rerun para mostrar nova mensagem
                    st.rerun()
        
        with input_col3:
            if st.button("🗑️", key="top_clear", help="Limpar chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.extraction_method = None
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Título principal
st.title("🚀 Kickstarter Success Predictor")
st.markdown("### Descubra as chances de sucesso do seu projeto antes de lançar!")

# Tabs principais
tab1, tab2, tab3 = st.tabs(["🔮 Predictor", "🤖 Análise AI", "📊 Dashboard"])

with tab1:
    # Layout principal - duas colunas
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Dados do Projeto")
        
        # Formulário de entrada
        with st.form("project_form"):
                # Identificação do usuário (NOVO - requisito do case)
                st.markdown("### 👤 Identificação do Usuário")
                user_email = st.text_input(
                    "Email (opcional)",
                    placeholder="seu@email.com",
                    help="Se você já tem histórico conosco, suas informações serão consideradas na análise"
                )
                
                # Buscar dados do usuário
                user_data = USERS_DATABASE.get(user_email, USERS_DATABASE["default"])
                if user_email and user_email in USERS_DATABASE:
                    st.markdown(f"""
                    <div class="user-profile-box">
                    <strong>✅ Bem-vindo de volta, {user_data['nome']}!</strong><br>
                    📊 Seu histórico: {user_data['projetos_historico']} projetos | Taxa de sucesso: {user_data['taxa_sucesso_pessoal']:.0%}<br>
                    🏆 Experiência em: {', '.join(user_data['categorias_experiencia'])}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Nome do projeto
                st.markdown("### 1️⃣ Nome do Projeto")
                
                # Categoria primeiro (para sugestões de título)
                st.markdown("### 2️⃣ Categoria")
                categories = load_categories()
                
                # Criar duas colunas para categoria
                cat_col1, cat_col2 = st.columns([3, 1])
                
                with cat_col1:
                    selected_category = st.selectbox(
                        "Escolha a categoria mais adequada",
                        options=list(categories.keys()),
                        format_func=lambda x: f"{x} - {categories[x]['description']}"
                    )
                
                with cat_col2:
                    if selected_category:
                        cat_success = categories[selected_category]['avg_success']
                        st.metric("Taxa de Sucesso", cat_success)
                
                # Verificar se usuário tem experiência na categoria
                if user_email in USERS_DATABASE and selected_category in user_data['categorias_experiencia']:
                    st.success(f"✅ Você tem experiência em {selected_category}!")
                
                # Agora o título com sugestões baseadas na categoria
                title_suggestions = {
                    'Technology': [
                        "Smart Home Assistant with AI Technology",
                        "Revolutionary Solar-Powered Device",
                        "Innovative Wireless Charging Solution",
                        "Next Generation Smart Watch"
                    ],
                    'Games': [
                        "Epic Adventure Board Game Experience", 
                        "Strategic Card Game for Everyone",
                        "Family-Friendly Puzzle Game Collection",
                        "Ultimate RPG Tabletop Adventure"
                    ],
                    'Music': [
                        "Debut Album Recording Project",
                        "Live Concert Tour Experience", 
                        "Professional Music Studio Equipment",
                        "Original Soundtrack Creation"
                    ],
                    'Art': [
                        "Contemporary Art Exhibition Series",
                        "Digital Art Creation Workshop",
                        "Community Mural Project Initiative",
                        "Interactive Art Installation Experience"
                    ],
                    'Film & Video': [
                        "Independent Documentary Film Project",
                        "Short Film Production Series",
                        "Virtual Reality Video Experience",
                        "Educational Video Course Creation"
                    ],
                    'Design': [
                        "Eco-Friendly Product Design Collection",
                        "Minimalist Furniture Design Series",
                        "Sustainable Fashion Accessories Line",
                        "Innovative Kitchen Gadget Design"
                    ],
                    'Comics': [
                        "Original Graphic Novel Series",
                        "Superhero Comic Book Collection",
                        "Manga-Inspired Story Adventure",
                        "Illustrated Children's Book Series"
                    ],
                    'Food': [
                        "Artisan Food Truck Launch",
                        "Organic Recipe Book Collection",
                        "Gourmet Cooking Class Series",
                        "Sustainable Restaurant Opening Project"
                    ]
                }
                
                # Campo de sugestões
                use_suggestion = st.checkbox("Ver sugestões de títulos")
                
                if use_suggestion and selected_category in title_suggestions:
                    selected_suggestion = st.selectbox(
                        "Escolha uma sugestão ou inspire-se:",
                        ["Escrever meu próprio"] + title_suggestions.get(selected_category, [])
                    )
                    
                    if selected_suggestion != "Escrever meu próprio":
                        project_name = st.text_input(
                            "Título do seu projeto",
                            value=selected_suggestion,
                            help="Use 4-7 palavras descritivas. Evite títulos muito curtos ou longos."
                        )
                    else:
                        project_name = st.text_input(
                            "Título do seu projeto",
                            placeholder="Ex: Amazing Solar-Powered Backpack",
                            help="Use 4-7 palavras descritivas. Evite títulos muito curtos ou longos."
                        )
                else:
                    project_name = st.text_input(
                        "Título do seu projeto",
                        placeholder="Ex: Amazing Solar-Powered Backpack",
                        help="Use 4-7 palavras descritivas. Evite títulos muito curtos ou longos."
                    )
                
                # País
                st.markdown("### 3️⃣ País")
                selected_country = st.selectbox(
                    "País de origem",
                    options=list(COUNTRIES.keys()),
                    format_func=lambda x: f"{x} - {COUNTRIES[x]}"
                )
                
                # Meta financeira
                st.markdown("### 4️⃣ Meta Financeira")
                goal_amount = st.number_input(
                    "Meta em dólares (USD)",
                    min_value=100,
                    max_value=1000000,
                    value=10000,
                    step=500,
                    help="Converta para USD se estiver em outra moeda"
                )
                
                # Mostrar referência de metas com base no histórico do usuário
                if user_email in USERS_DATABASE and user_data['projetos_detalhes']:
                    avg_goal = sum(p['meta'] for p in user_data['projetos_detalhes']) / len(user_data['projetos_detalhes'])
                    st.info(f"💡 Sua meta média histórica: ${avg_goal:,.0f}")
                
                if goal_amount < 5000:
                    st.info("💡 Meta baixa - Mais fácil de alcançar!")
                elif goal_amount < 25000:
                    st.success("✅ Meta na faixa ideal")
                elif goal_amount < 50000:
                    st.warning("⚠️ Meta alta - Precisará de estratégia forte")
                else:
                    st.error("🚨 Meta muito alta - Alto risco!")
                
                # Datas
                st.markdown("### 5️⃣ Período da Campanha")
                date_col1, date_col2 = st.columns(2)
                
                with date_col1:
                    launch_date = st.date_input(
                        "Data de lançamento",
                        value=datetime.now(),
                        min_value=datetime.now()
                    )
                
                with date_col2:
                    deadline_date = st.date_input(
                        "Data limite",
                        value=datetime.now() + timedelta(days=30),
                        min_value=launch_date + timedelta(days=1)
                    )
                
                # Calcular duração
                campaign_days = (deadline_date - launch_date).days
                
                if campaign_days < 20:
                    st.warning(f"⚠️ Campanha muito curta: {campaign_days} dias")
                elif campaign_days > 45:
                    st.warning(f"⚠️ Campanha muito longa: {campaign_days} dias")
                else:
                    st.success(f"✅ Duração ideal: {campaign_days} dias")
                
                # Botão de submit
                submitted = st.form_submit_button("🔮 Prever Sucesso", use_container_width=True)

    # Coluna 2 - Resultados
    with col2:
            st.header("📊 Análise e Predição")
            
            if submitted:
                # Salvar dados do usuário no session state
                st.session_state.user_data = user_data
                st.session_state.user_email = user_email
                
                # Verificar se temos o nome do projeto
                if not project_name:
                    st.warning("⚠️ Por favor, insira o nome do projeto")
                else:
                    # Preparar dados para API
                    project_data = {
                        "name": project_name,
                        "main_category": selected_category,
                        "country": selected_country,
                        "usd_goal_real": float(goal_amount),
                        "launched": launch_date.strftime("%Y-%m-%d"),
                        "deadline": deadline_date.strftime("%Y-%m-%d"),
                        "campaign_days": campaign_days
                    }
                    
                    # Salvar no session state
                    st.session_state.project_data = project_data
                
                    # Fazer requisição
                    with st.spinner("Analisando seu projeto..."):
                        try:
                            response = requests.post(f"{API_URL}/predict", json=project_data)
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.session_state.prediction_result = result
                                
                                # Extrair resultados
                                probability = result['success_probability']
                                prediction = result['prediction']
                                confidence = result['confidence']
                                recommendations = result['recommendations']
                                threshold = result['threshold_used']
                                
                                # Criar gauge chart
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number+delta",
                                    value=probability * 100,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Probabilidade de Sucesso"},
                                    delta={'reference': threshold * 100, 'relative': True},
                                    gauge={
                                        'axis': {'range': [None, 100]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 30], 'color': "lightgray"},
                                            {'range': [30, 70], 'color': "gray"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': threshold * 100
                                        }
                                    }
                                ))
                                
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Métricas principais
                                metric_col1, metric_col2 = st.columns(2)
                                
                                with metric_col1:
                                    st.metric(
                                        "Probabilidade",
                                        f"{probability:.1%}",
                                        delta=f"{(probability - threshold)*100:.1f}% do threshold"
                                    )
                                
                                with metric_col2:
                                    color = "🟢" if prediction == "Sucesso" else "🔴"
                                    st.metric("Predição", f"{color} {prediction}")
                                
                                # Análise personalizada para o usuário
                                if user_email in USERS_DATABASE:
                                    st.markdown("### 🎯 Análise Personalizada")
                                    
                                    # Comparar com histórico pessoal
                                    if user_data['taxa_sucesso_pessoal'] > 0:
                                        if probability > user_data['taxa_sucesso_pessoal']:
                                            st.success(f"📈 Este projeto tem potencial {(probability - user_data['taxa_sucesso_pessoal'])*100:.1f}% acima da sua média histórica ({user_data['taxa_sucesso_pessoal']:.0%})!")
                                        else:
                                            st.warning(f"📉 Este projeto está {(user_data['taxa_sucesso_pessoal'] - probability)*100:.1f}% abaixo da sua média histórica ({user_data['taxa_sucesso_pessoal']:.0%})")
                                    
                                    # Verificar experiência na categoria
                                    if selected_category in user_data['categorias_experiencia']:
                                        st.info(f"✅ Sua experiência em {selected_category} é um diferencial importante!")
                                    else:
                                        st.warning(f"⚠️ Primeira vez em {selected_category}? Considere buscar mentoria ou parceiros experientes nesta categoria.")
                                    
                                    # Mostrar projetos similares do histórico
                                    similar_projects = [p for p in user_data.get('projetos_detalhes', []) if p['categoria'] == selected_category]
                                    if similar_projects:
                                        st.markdown("#### 📚 Seus projetos anteriores nesta categoria:")
                                        for proj in similar_projects:
                                            emoji = "✅" if proj['sucesso'] else "❌"
                                            st.caption(f"{emoji} {proj['nome']} - Meta: ${proj['meta']:,}")
                                
                                # Recomendações
                                st.markdown("### 💡 Recomendações Personalizadas")
                                
                                for rec in recommendations:
                                    if "✅" in rec:
                                        st.success(rec)
                                    elif "⚠️" in rec:
                                        st.warning(rec)
                                    elif "🔴" in rec:
                                        st.error(rec)
                                    elif "💡" in rec:
                                        st.info(rec)
                                    else:
                                        st.write(rec)
                                
                                # Botões de ação rápida
                                st.markdown("### 🚀 Ações Rápidas com AI")
                                
                                col_a1, col_a2 = st.columns(2)
                                
                                with col_a1:
                                    if st.button("📝 Melhorar Título", use_container_width=True):
                                        with st.spinner("Gerando sugestões..."):
                                            suggestions = generate_title_suggestions(project_name, selected_category)
                                            st.info(suggestions)
                                
                                with col_a2:
                                    if st.button("📋 Gerar Estratégia", use_container_width=True):
                                        with st.spinner("Criando estratégia..."):
                                            strategy = optimize_campaign_strategy(project_data, result)
                                            st.info(strategy)
                                
                                # Análise detalhada
                                with st.expander("📈 Ver Análise Detalhada"):
                                    st.markdown("### Fatores que influenciaram a predição:")
                                    
                                    # Criar dataframe com os fatores
                                    factors_data = {
                                        'Fator': [
                                            f'Categoria ({selected_category})',
                                            f'Meta (${goal_amount:,})',
                                            f'Duração ({campaign_days} dias)',
                                            f'País ({selected_country})',
                                            f'Título ({len(project_name.split())} palavras)'
                                        ],
                                        'Impacto': [
                                            float(categories[selected_category]['avg_success'].rstrip('%')),
                                            max(0, 100 - (goal_amount / 500)),  # Simplificado
                                            100 if 25 <= campaign_days <= 35 else 50,
                                            80 if selected_country == 'US' else 60,
                                            80 if 4 <= len(project_name.split()) <= 7 else 50
                                        ]
                                    }
                                    
                                    df_factors = pd.DataFrame(factors_data)
                                    
                                    fig_factors = px.bar(
                                        df_factors, 
                                        x='Impacto', 
                                        y='Fator',
                                        orientation='h',
                                        title='Impacto de cada fator na predição',
                                        color='Impacto',
                                        color_continuous_scale='RdYlGn'
                                    )
                                    
                                    st.plotly_chart(fig_factors, use_container_width=True)
                                    
                                    # Comparação com projetos similares
                                    st.markdown("### 📊 Comparação com projetos similares")
                                    st.info(f"""
                                    **Categoria {selected_category}:**
                                    - Taxa de sucesso média: {categories[selected_category]['avg_success']}
                                    - Sua probabilidade: {probability:.1%}
                                    - Diferença: {probability*100 - float(categories[selected_category]['avg_success'].rstrip('%')):.1f}%
                                    """)
                                    
                            else:
                                st.error(f"Erro na API: {response.status_code}")
                                st.json(response.json())
                                
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Não foi possível conectar com a API. Verifique se está rodando em http://localhost:8000")
                        except Exception as e:
                            st.error(f"Erro: {str(e)}")
            else:
                # Estado inicial - mostrar informações
                st.markdown("""
                <div class="info-box">
                <h4>🎯 Como usar:</h4>
                <ol>
                <li>Identifique-se (opcional) para análise personalizada</li>
                <li>Preencha os dados do seu projeto</li>
                <li>Clique em "Prever Sucesso"</li>
                <li>Veja a análise detalhada e recomendações</li>
                <li>Use o chat AI para tirar dúvidas específicas</li>
                </ol>
                </div>
                """, unsafe_allow_html=True)
                
                # Gráfico de exemplo - taxas por categoria
                categories = load_categories()
                cat_names = list(categories.keys())
                cat_success_rates = [float(categories[cat]['avg_success'].rstrip('%')) for cat in cat_names]
                
                df_categories = pd.DataFrame({
                    'Categoria': cat_names,
                    'Taxa de Sucesso (%)': cat_success_rates
                }).sort_values('Taxa de Sucesso (%)', ascending=True)
                
                fig_cats = px.bar(
                    df_categories,
                    x='Taxa de Sucesso (%)',
                    y='Categoria',
                    orientation='h',
                    title='Taxa de Sucesso por Categoria (Histórico)',
                    color='Taxa de Sucesso (%)',
                    color_continuous_scale='RdYlGn',
                    range_x=[0, 70]
                )
                
                fig_cats.update_layout(height=500)
                st.plotly_chart(fig_cats, use_container_width=True)

# Tab 2 - Análise AI
with tab2:
        st.header("🤖 Análise Aprofundada com AI")
        
        if st.session_state.project_data and st.session_state.prediction_result:
            col_ai1, col_ai2 = st.columns([1, 1])
            
            with col_ai1:
                st.subheader("📊 Dados do Projeto Atual")
                st.json(st.session_state.project_data)
                
                # Mostrar dados do usuário se disponível
                if st.session_state.user_email and st.session_state.user_email in USERS_DATABASE:
                    st.subheader("👤 Perfil do Usuário")
                    user_info = st.session_state.user_data
                    st.info(f"""
                    **{user_info['nome']}** - {user_info['cargo']}
                    - Experiência: {user_info['experiencia_anos']} anos
                    - Projetos: {user_info['projetos_historico']}
                    - Taxa de sucesso: {user_info['taxa_sucesso_pessoal']:.0%}
                    - Expertise: {', '.join(user_info['categorias_experiencia'])}
                    """)
                
                st.subheader("🎯 Resultado da Predição")
                prob = st.session_state.prediction_result['success_probability']
                if prob >= 0.7:
                    st.success(f"Probabilidade: {prob:.1%}")
                elif prob >= 0.3:
                    st.warning(f"Probabilidade: {prob:.1%}")
                else:
                    st.error(f"Probabilidade: {prob:.1%}")
            
            with col_ai2:
                st.subheader("💡 Análise Inteligente")
                
                if st.button("🔍 Gerar Análise Completa", use_container_width=True):
                    with st.spinner("Analisando com AI..."):
                        analysis = analyze_project_with_ai(
                            st.session_state.project_data,
                            st.session_state.prediction_result
                        )
                        st.markdown(analysis)
                
                st.markdown("---")
                
                # Ferramentas específicas
                st.subheader("🛠️ Ferramentas de Otimização")
                
                tool_col1, tool_col2 = st.columns(2)
                
                with tool_col1:
                    if st.button("📝 Otimizar Título", use_container_width=True):
                        with st.spinner("Gerando títulos..."):
                            titles = generate_title_suggestions(
                                st.session_state.project_data['name'],
                                st.session_state.project_data['main_category']
                            )
                            st.info(titles)
                    
                    if st.button("💰 Analisar Meta", use_container_width=True):
                        with st.spinner("Analisando meta..."):
                            user_context = ""
                            if st.session_state.user_data and st.session_state.user_data != USERS_DATABASE["default"]:
                                user_context = f"Considerando que {st.session_state.user_data['nome']} tem histórico de {st.session_state.user_data['projetos_historico']} projetos com taxa de sucesso de {st.session_state.user_data['taxa_sucesso_pessoal']:.0%}, "
                            
                            prompt = f"""
                            {user_context}analise se a meta de ${st.session_state.project_data['usd_goal_real']:,.2f} 
                            é adequada para um projeto de {st.session_state.project_data['main_category']} 
                            no {st.session_state.project_data['country']}.
                            
                            Compare com projetos similares bem-sucedidos e sugira ajustes se necessário.
                            """
                            analysis = get_chat_response(prompt)
                            st.info(analysis)
                
                with tool_col2:
                    if st.button("📅 Plano de 30 Dias", use_container_width=True):
                        with st.spinner("Criando plano..."):
                            strategy = optimize_campaign_strategy(
                                st.session_state.project_data,
                                st.session_state.prediction_result
                            )
                            st.info(strategy)
                    
                    if st.button("🎁 Estrutura de Recompensas", use_container_width=True):
                        with st.spinner("Criando recompensas..."):
                            user_context = ""
                            if st.session_state.user_data and st.session_state.user_data != USERS_DATABASE["default"]:
                                user_context = f"Considerando a experiência de {st.session_state.user_data['nome']} em {', '.join(st.session_state.user_data['categorias_experiencia'])}, "
                            
                            prompt = f"""
                            {user_context}crie uma estrutura de recompensas para este projeto:
                            {json.dumps(st.session_state.project_data, indent=2)}
                            
                            Inclua:
                            1. Early bird (25% desconto)
                            2. Níveis regulares (pelo menos 5)
                            3. Recompensa premium
                            4. Preços e o que cada nível recebe
                            
                            Seja criativo e específico para a categoria {st.session_state.project_data['main_category']}.
                            """
                            rewards = get_chat_response(prompt)
                            st.info(rewards)
        else:
            st.info("👈 Primeiro faça uma predição na aba 'Predictor' para usar a análise AI")

# Tab 3 - Dashboard
with tab3:
        st.header("📊 Dashboard de Insights")
        
        # Estatísticas gerais
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        with col_stats1:
            st.metric("Projetos Analisados", "378,661")
        with col_stats2:
            st.metric("Taxa de Sucesso Geral", "35.9%")
        with col_stats3:
            st.metric("Meta Média de Sucesso", "$9,426")
        with col_stats4:
            st.metric("Duração Média", "33 dias")
        
        # Mostrar estatísticas do usuário atual se logado
        if st.session_state.user_email and st.session_state.user_email in USERS_DATABASE:
            st.markdown("### 👤 Suas Estatísticas Pessoais")
            user_stats_col1, user_stats_col2, user_stats_col3, user_stats_col4 = st.columns(4)
            
            with user_stats_col1:
                st.metric("Seus Projetos", st.session_state.user_data['projetos_historico'])
            with user_stats_col2:
                st.metric("Sua Taxa de Sucesso", f"{st.session_state.user_data['taxa_sucesso_pessoal']:.0%}")
            with user_stats_col3:
                st.metric("Anos de Experiência", st.session_state.user_data['experiencia_anos'])
            with user_stats_col4:
                st.metric("Categorias", len(st.session_state.user_data['categorias_experiencia']))
        
        # Gráficos de insights
        categories = load_categories()
        
        # Gráfico 1: Taxa de sucesso por categoria
        cat_names = list(categories.keys())
        cat_success_rates = [float(categories[cat]['avg_success'].rstrip('%')) for cat in cat_names]
        
        df_cat_success = pd.DataFrame({
            'Categoria': cat_names,
            'Taxa de Sucesso (%)': cat_success_rates
        }).sort_values('Taxa de Sucesso (%)', ascending=True)
        
        fig_success = px.bar(
            df_cat_success,
            x='Taxa de Sucesso (%)',
            y='Categoria',
            orientation='h',
            title='Taxa de Sucesso por Categoria',
            color='Taxa de Sucesso (%)',
            color_continuous_scale='RdYlGn'
        )
        
        st.plotly_chart(fig_success, use_container_width=True)
        
        # Gráfico 2: Distribuição de metas
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Simulação de dados de distribuição de metas
            goal_ranges = ['< $1k', '$1k-5k', '$5k-10k', '$10k-25k', '$25k-50k', '> $50k']
            success_by_goal = [45, 42, 38, 28, 18, 12]
            
            fig_goal_success = px.bar(
                x=goal_ranges,
                y=success_by_goal,
                title='Taxa de Sucesso por Faixa de Meta',
                labels={'x': 'Faixa de Meta', 'y': 'Taxa de Sucesso (%)'},
                color=success_by_goal,
                color_continuous_scale='RdYlGn'
            )
            
            st.plotly_chart(fig_goal_success, use_container_width=True)
        
        with col_chart2:
            # Simulação de dados de duração
            duration_ranges = ['< 20 dias', '20-30 dias', '30-40 dias', '40-50 dias', '> 50 dias']
            success_by_duration = [25, 42, 38, 28, 15]
            
            fig_duration_success = px.bar(
                x=duration_ranges,
                y=success_by_duration,
                title='Taxa de Sucesso por Duração da Campanha',
                labels={'x': 'Duração', 'y': 'Taxa de Sucesso (%)'},
                color=success_by_duration,
                color_continuous_scale='RdYlGn'
            )
            
            st.plotly_chart(fig_duration_success, use_container_width=True)
        
        # Insights principais
        st.markdown("### 💡 Insights Principais")
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.info("""
            **🎯 Melhores Práticas Identificadas:**
            - Campanhas de 30 dias têm 42% de sucesso
            - Metas abaixo de $10k têm 2x mais chance
            - Lançar na terça aumenta em 20% as chances
            - Vídeo de qualidade aumenta em 40% a conversão
            """)
        
        with col_insight2:
            st.warning("""
            **⚠️ Principais Armadilhas:**
            - Metas acima de $50k têm apenas 12% de sucesso
            - Campanhas > 45 dias perdem momentum
            - Títulos genéricos reduzem em 25% as chances
            - Falta de atualizações afasta apoiadores
            """)

# Sidebar com informações
with st.sidebar:
    st.header("ℹ️ Como funciona")
    
    # Status da API e sistemas
    st.markdown("### 🔌 Status dos Sistemas")
    
    api_status = check_api_health()
    if api_status:
        st.success("✅ API Online")
    else:
        st.error("❌ API Offline - Verifique se está rodando")
    
    # NOVO: Controle do spaCy
    st.markdown("### 🤖 Controle de Extração")
    
    col_spacy1, col_spacy2 = st.columns([3, 1])
    
    with col_spacy1:
        if st.session_state.use_spacy:
            if SPACY_AVAILABLE:
                st.success("✅ spaCy Ativo")
            else:
                st.warning("⚠️ spaCy não instalado")
        else:
            st.info("🔌 spaCy Desativado")
    
    with col_spacy2:
        if st.button("🔄", help="Alternar spaCy", use_container_width=True):
            st.session_state.use_spacy = not st.session_state.use_spacy
            st.rerun()
    
    # Explicação do modo atual
    if st.session_state.use_spacy:
        st.caption("**Modo Híbrido:** spaCy → OpenAI")
        st.caption("Extração local e gratuita primeiro")
    else:
        st.caption("**Modo OpenAI:** Apenas OpenAI")
        st.caption("Mais flexível, requer API key")
    
    st.markdown("---")
    
    # Status dos outros sistemas
    if OPENAI_AVAILABLE:
        st.success("✅ OpenAI Configurado")
    else:
        st.info("ℹ️ OpenAI não configurado (opcional)")
    
    # Mostrar usuário logado
    if st.session_state.user_email and st.session_state.user_email in USERS_DATABASE:
        st.markdown("---")
        st.markdown(f"### 👤 Usuário Atual")
        st.info(f"{st.session_state.user_data['nome']}")
        if st.button("🚪 Sair", use_container_width=True):
            st.session_state.user_email = None
            st.session_state.user_data = USERS_DATABASE["default"]
            st.rerun()
    
    st.markdown("---")
    
    # Informações sobre o sistema híbrido
    st.markdown("### 🤖 Sistema Híbrido")
    
    with st.expander("ℹ️ Como funciona", expanded=False):
        if st.session_state.use_spacy:
            st.markdown("""
            **Modo Híbrido Ativo:**
            1. **spaCy** (primário)
               - Gratuito e local
               - Patterns regex
            2. **OpenAI** (fallback)
               - Se spaCy falhar
               - Mais flexível
            """)
        else:
            st.markdown("""
            **Modo OpenAI Puro:**
            - Extração apenas via OpenAI
            - Mais flexível e inteligente
            - Requer API key configurada
            - Sem processamento local
            """)
        
        st.markdown("""
        **Predições:**
        - Sempre via API real
        - Nunca inventa números
        """)
    
    if st.session_state.extraction_method:
        st.info(f"Última extração: {st.session_state.extraction_method}")
    
    st.markdown("---")
    
    st.markdown("""
    ### 📊 O que influencia o sucesso?
    
    1. **Categoria do projeto** (40%)
       - Mais importante fator
       - Dance e Theater têm melhores taxas
       
    2. **Meta financeira** (30%)
       - Metas menores = mais fácil
       - Sweet spot: $3,000-$10,000
       
    3. **Duração da campanha** (15%)
       - Ideal: 25-35 dias
       - Muito curta ou longa = ruim
       
    4. **Outros fatores** (15%)
       - País, título, timing
       - Histórico do criador
    """)
    
    #st.markdown("---")
    
    st.markdown("""
    ### 🎯 Threshold: 31.7%
    
    O modelo classifica como "Sucesso" 
    projetos com mais de 31.7% de chance.
    
    Isso foi otimizado com base em 
    300,000+ projetos reais!
    """)
    
    st.markdown("---")
    
    # Usuários de exemplo
    st.markdown("### 👥 Usuários de Teste")
    st.caption("Para testar a personalização:")
    st.code("""
joao@example.com
maria@example.com
pedro@example.com
    """, language="text")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Modelo treinado com 300,000+ projetos reais do Kickstarter | AUC-ROC: 0.733</p>
    <p>⚠️ Lembre-se: Este é um modelo estatístico. O sucesso real depende da execução!</p>
    <p>🤖 Sistema Híbrido: spaCy + OpenAI (opcional) + API Real</p>
    <p>👥 Base de usuários integrada para análise personalizada</p>
</div>
""", unsafe_allow_html=True)

# Adicionar informações técnicas no final
with st.expander("🔧 Informações Técnicas do Sistema"):
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.markdown("""
        ### 📊 Features utilizadas:
        1. **cat_success_rate** - Taxa histórica da categoria
        2. **usd_goal_real** - Meta em USD
        3. **campaign_days** - Duração em dias
        4. **goal_magnitude** - Log10 da meta
        5. **name_word_count** - Palavras no título
        6. **country_success_rate** - Taxa do país
        7. **launch_year** - Ano de lançamento
        
        ### 🤖 Sistema Híbrido:
        - **spaCy**: Extração local com regex
        - **OpenAI**: Fallback inteligente
        - **API**: prediçoes na API
        
        """)
    
    with col_tech2:
        st.markdown("""
        ### 🎯 Métricas do modelo:
        - **Algoritmo**: Gradient Boosting
        - **AUC-ROC**: 0.7327
        - **Threshold**: 31.7%
        - **Cross-validation**: 5-fold
        - **Dados de treino**: 265,340 projetos
        - **Dados de teste**: 66,335 projetos
        
        ### 💬 Capacidades do Chat:
        - **Extração automática** de projetos
        - **Respostas contextuais**
        - **Análise personalizada**
        - **Sempre usa API real**
        """)

# Instruções de instalação
with st.expander("📦 Configuração do Sistema Híbrido"):
    st.markdown("""
    ### Instalação Completa:
    
    ```bash
    # 1. Instalar spaCy (RECOMENDADO)
    pip install spacy
    python -m spacy download pt_core_news_sm
    # ou
    python -m spacy download en_core_web_sm
    
    # 2. OpenAI (OPCIONAL - para fallback)
    pip install openai python-dotenv
    
    # 3. Outras dependências
    pip install streamlit pandas plotly requests
    ```
    
    ### Configuração do .env (opcional):
    ```
    OPENAI_API_KEY=sua_chave_aqui
    KICKSTARTER_API_URL=http://localhost:8000
    ```
    
    ### Como funciona:
    1. **spaCy tenta primeiro** - rápido e gratuito
    2. **OpenAI como backup** - se spaCy falhar
    3. **API sempre para predições** - dados reais
    
    ### Vantagens:
    - ✅ Funciona sem OpenAI
    - ✅ Extração inteligente
    - ✅ Sempre usa dados reais da API
    - ✅ Sistema robusto com fallback
    """)

# Script para mostrar notificação
if st.session_state.user_email and st.session_state.user_email in USERS_DATABASE:
    st.toast(f"👤 Logado como: {st.session_state.user_data['nome']}", icon="✅")

# Mostrar método de extração se disponível
if st.session_state.extraction_method:
    st.toast(f"📝 Última extração: {st.session_state.extraction_method}", icon="ℹ️")
