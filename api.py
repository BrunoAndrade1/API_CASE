"""
API FastAPI para o modelo Kickstarter Success Predictor

Para executar em produção (ex: no Render):
1. Certifique-se que o arquivo 'kickstarter_model_v1.pkl' e 'ml_classes.py' existem.
2. Use o comando de início: uvicorn api:app --host 0.0.0.0 --port 10000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os
import requests
from ml_classes import KickstarterPreprocessor, KickstarterPredictor

# Importações adicionais para treinamento
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURAÇÃO DA API
# =====================================================

app = FastAPI(
    title="Kickstarter Success Predictor API",
    description="API para prever sucesso de projetos no Kickstarter usando Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique os domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MODELOS DE DADOS (SCHEMAS)
# =====================================================

class ProjectInput(BaseModel):
    """Schema para entrada de dados de um projeto"""
    
    name: str = Field(
        ..., 
        description="Nome/título do projeto",
        example="Amazing Solar-Powered Backpack"
    )
    
    main_category: str = Field(
        ...,
        description="Categoria principal do projeto",
        example="Technology"
    )
    
    country: str = Field(
        ...,
        description="Código do país (2 letras)",
        example="US"
    )
    
    usd_goal_real: float = Field(
        ...,
        description="Meta em dólares americanos (USD)",
        example=15000.0,
        gt=0,
        le=100000000
    )
    
    launched: str = Field(
        ...,
        description="Data de lançamento (YYYY-MM-DD)",
        example="2024-03-01"
    )
    
    deadline: str = Field(
        ...,
        description="Data limite (YYYY-MM-DD)",
        example="2024-03-31"
    )
    
    @validator('country')
    def validate_country(cls, v):
        if len(v) != 2:
            raise ValueError('País deve ter código de 2 letras (ex: US, GB, BR)')
        return v.upper()
    
    @validator('main_category')
    def validate_category(cls, v):
        valid_categories = [
            'Film & Video', 'Music', 'Publishing', 'Games', 'Technology',
            'Design', 'Art', 'Comics', 'Theater', 'Food', 'Photography',
            'Fashion', 'Dance', 'Journalism', 'Crafts'
        ]
        if v not in valid_categories:
            raise ValueError(f'Categoria inválida. Use uma das: {", ".join(valid_categories)}')
        return v
    
    @validator('deadline')
    def validate_dates(cls, v, values):
        if 'launched' in values:
            try:
                launched_date = datetime.strptime(values['launched'], '%Y-%m-%d')
                deadline_date = datetime.strptime(v, '%Y-%m-%d')
                
                if deadline_date <= launched_date:
                    raise ValueError('Deadline deve ser após a data de lançamento')
                
                days_diff = (deadline_date - launched_date).days
                if days_diff > 365:
                    raise ValueError('Campanha não pode durar mais de 365 dias')
                if days_diff < 1:
                    raise ValueError('Campanha deve durar pelo menos 1 dia')
                    
            except ValueError as e:
                if "time data" in str(e):
                    raise ValueError('Data deve estar no formato YYYY-MM-DD')
                raise
        return v


class PredictionOutput(BaseModel):
    """Schema para resposta da predição"""
    
    success_probability: float = Field(..., description="Probabilidade de sucesso (0.0 a 1.0)")
    prediction: str = Field(..., description="Predição final: 'Sucesso' ou 'Falha'")
    confidence: str = Field(..., description="Nível de confiança: 'Alta', 'Média' ou 'Baixa'")
    recommendations: List[str] = Field(..., description="Lista de recomendações personalizadas")
    threshold_used: float = Field(..., description="Threshold usado para classificação")
    
    class Config:
        schema_extra = {
            "example": {
                "success_probability": 0.743,
                "prediction": "Sucesso",
                "confidence": "Alta",
                "recommendations": [
                    "✅ Meta dentro da faixa recomendada.",
                    "✅ Duração adequada da campanha.",
                    "🌟 Excelentes chances! Foque na execução."
                ],
                "threshold_used": 0.317
            }
        }


class BatchInput(BaseModel):
    """Schema para predição em lote"""
    projects: List[ProjectInput]


class ModelInfo(BaseModel):
    """Schema para informações do modelo"""
    version: str
    training_date: str
    metrics: dict
    features_used: List[str]
    threshold: float


class HealthCheck(BaseModel):
    """Schema para health check"""
    status: str
    model_loaded: bool
    timestamp: str


# =====================================================
# VARIÁVEIS GLOBAIS E FUNÇÕES AUXILIARES
# =====================================================

MODEL_PATH = 'kickstarter_model_v1.pkl'
DATA_URL = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0114ENv3/Dataset/kickstarter_grading.csv'
model_data = None
predictor = None
training_in_progress = False

def download_data():
    """Baixa os dados do Kickstarter se não existirem"""
    print("Baixando dados do Kickstarter...")
    try:
        response = requests.get(DATA_URL)
        with open('kickstarter_data.csv', 'wb') as f:
            f.write(response.content)
        print("✓ Dados baixados com sucesso!")
        return True
    except Exception as e:
        print(f"Erro ao baixar dados: {e}")
        return False

def train_model_async():
    """Treina o modelo de forma assíncrona"""
    global model_data, predictor, training_in_progress
    
    training_in_progress = True
    
    try:
        print("\n" + "="*80)
        print("INICIANDO TREINAMENTO DO MODELO")
        print("="*80)
        
        # Baixar dados se necessário
        if not os.path.exists('kickstarter_data.csv'):
            if not download_data():
                raise Exception("Falha ao baixar dados")
        
        # Carregar dados
        print("\n[1/7] Carregando dados...")
        df = pd.read_csv('kickstarter_data.csv', encoding='latin-1')
        df.columns = df.columns.str.strip()
        print(f"✓ Dados carregados: {len(df):,} projetos")
        
        # Filtrar projetos finalizados
        print("\n[2/7] Filtrando projetos...")
        df = df[df['state'].isin(['failed', 'successful'])]
        df['success'] = (df['state'] == 'successful').astype(int)
        print(f"✓ Projetos válidos: {len(df):,}")
        
        # Dividir dados
        print("\n[3/7] Dividindo dados...")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['success'])
        
        # Criar e ajustar preprocessador
        print("\n[4/7] Criando preprocessador...")
        preprocessor = KickstarterPreprocessor()
        preprocessor.fit(train_df)
        
        # Transformar dados
        print("\n[5/7] Transformando dados...")
        X_train = preprocessor.transform(train_df)
        X_test = preprocessor.transform(test_df)
        y_train = train_df['success'].values
        y_test = test_df['success'].values
        
        # Treinar modelo
        print("\n[6/7] Treinando modelo...")
        model = GradientBoostingClassifier(
            n_estimators=100,      # Reduzido para treinar mais rápido
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Avaliar modelo
        print("\n[7/7] Avaliando modelo...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Encontrar threshold ótimo
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # Salvar modelo
        model_data = {
            'model': model,
            'preprocessor': preprocessor,
            'optimal_threshold': optimal_threshold,
            'feature_names': preprocessor.features_selected,
            'version': '1.0',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'auc_roc': auc_roc,
                'n_train': len(train_df),
                'n_test': len(test_df)
            }
        }
        
        joblib.dump(model_data, MODEL_PATH)
        
        # Criar predictor
        predictor = KickstarterPredictor(
            model=model_data['model'],
            preprocessor=model_data['preprocessor'],
            threshold=model_data['optimal_threshold']
        )
        
        print(f"\n✓ Modelo treinado com sucesso!")
        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  Threshold: {optimal_threshold:.3f}")
        
    except Exception as e:
        print(f"\n❌ Erro no treinamento: {e}")
        
    finally:
        training_in_progress = False

def load_model():
    """Carrega o modelo do disco"""
    global model_data, predictor
    
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Modelo não encontrado. Use o endpoint /train para treinar um novo modelo.")
        return False
    
    print(f"Carregando modelo de '{MODEL_PATH}'...")
    model_data = joblib.load(MODEL_PATH)
    
    predictor = KickstarterPredictor(
        model=model_data['model'],
        preprocessor=model_data['preprocessor'],
        threshold=model_data['optimal_threshold']
    )
    
    print(f"✓ Modelo carregado com sucesso!")
    print(f"  Versão: {model_data['version']}")
    print(f"  Treinado em: {model_data['training_date']}")
    print(f"  AUC-ROC: {model_data['metrics']['auc_roc']:.4f}")
    
    return True

# Tentar carregar modelo ao iniciar a aplicação
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"⚠️ Modelo não carregado na inicialização: {e}")
        print("Use o endpoint /train para treinar um novo modelo")

# =====================================================
# ENDPOINTS DA API
# =====================================================

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz com informações básicas"""
    return {
        "message": "Kickstarter Success Predictor API",
        "version": "1.0.0",
        "status": "online" if predictor else "modelo não carregado",
        "model_exists": os.path.exists(MODEL_PATH),
        "training_in_progress": training_in_progress,
        "endpoints": {
            "documentação_interativa": "/docs",
            "documentação_alternativa": "/redoc",
            "treinar_modelo": "/train",
            "fazer_predição": "/predict",
            "predição_em_lote": "/predict/batch",
            "informações_do_modelo": "/info/model",
            "categorias_válidas": "/info/categories",
            "países_suportados": "/info/countries",
            "health_check": "/health"
        }
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Verifica se a API e o modelo estão funcionando"""
    is_model_loaded = predictor is not None
    return {
        "status": "healthy" if is_model_loaded else "unhealthy",
        "model_loaded": is_model_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/train", tags=["Training"])
async def train_model(background_tasks: BackgroundTasks):
    """
    Inicia o treinamento de um novo modelo.
    O treinamento é feito em background.
    """
    global training_in_progress
    
    if training_in_progress:
        return {
            "status": "training_already_in_progress",
            "message": "Um treinamento já está em andamento"
        }
    
    # Adicionar tarefa de treinamento em background
    background_tasks.add_task(train_model_async)
    
    return {
        "status": "training_started",
        "message": "Treinamento iniciado em background. Use /health para verificar quando estiver pronto."
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_project(project: ProjectInput):
    """
    Faz predição para um único projeto Kickstarter.
    """
    if training_in_progress:
        raise HTTPException(
            status_code=503,
            detail="Modelo está sendo treinado. Aguarde alguns minutos."
        )
    
    if not predictor:
        raise HTTPException(
            status_code=503,
            detail="Modelo não está carregado. Use o endpoint /train para treinar um novo modelo."
        )
    
    try:
        project_data = project.dict()
        result = predictor.predict_single(project_data)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao fazer predição: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(batch: BatchInput):
    """
    Faz predição para múltiplos projetos de uma vez.
    """
    if training_in_progress:
        raise HTTPException(
            status_code=503,
            detail="Modelo está sendo treinado. Aguarde alguns minutos."
        )
    
    if not predictor:
        raise HTTPException(
            status_code=503,
            detail="Modelo não está carregado. Use o endpoint /train para treinar um novo modelo."
        )
    
    results = []
    
    for project in batch.projects:
        try:
            project_data = project.dict()
            result = predictor.predict_single(project_data)
            
            results.append({
                "project_name": project.name,
                "success": True,
                "result": result
            })
            
        except Exception as e:
            results.append({
                "project_name": project.name,
                "success": False,
                "error": str(e)
            })
    
    successful = sum(1 for r in results if r['success'])
    
    return {
        "total_projects": len(batch.projects),
        "successful_predictions": successful,
        "failed_predictions": len(batch.projects) - successful,
        "results": results
    }


@app.get("/info/model", response_model=ModelInfo, tags=["Information"])
async def get_model_info():
    """Retorna informações detalhadas sobre o modelo"""
    if not model_data:
        raise HTTPException(
            status_code=503,
            detail="Modelo não está carregado"
        )
    
    return {
        "version": model_data['version'],
        "training_date": model_data['training_date'],
        "metrics": model_data['metrics'],
        "features_used": model_data['feature_names'],
        "threshold": model_data['optimal_threshold']
    }


@app.get("/info/categories", tags=["Information"])
async def get_categories():
    """Lista todas as categorias válidas com estatísticas"""
    return {
        "total": 15,
        "categories": [
            {"value": "Film & Video", "description": "Filmes, documentários, vídeos", "avg_success": "42%"},
            {"value": "Music", "description": "Álbuns, shows, instrumentos", "avg_success": "53%"},
            {"value": "Publishing", "description": "Livros, revistas, e-books", "avg_success": "35%"},
            {"value": "Games", "description": "Jogos de tabuleiro, card games, RPG", "avg_success": "44%"},
            {"value": "Technology", "description": "Gadgets, apps, hardware", "avg_success": "24%"},
            {"value": "Design", "description": "Produtos, móveis, acessórios", "avg_success": "42%"},
            {"value": "Art", "description": "Pinturas, esculturas, instalações", "avg_success": "45%"},
            {"value": "Comics", "description": "HQs, graphic novels, mangás", "avg_success": "59%"},
            {"value": "Theater", "description": "Peças, musicais, performances", "avg_success": "64%"},
            {"value": "Food", "description": "Restaurantes, produtos alimentícios", "avg_success": "28%"},
            {"value": "Photography", "description": "Projetos fotográficos, livros de fotos", "avg_success": "34%"},
            {"value": "Fashion", "description": "Roupas, calçados, acessórios", "avg_success": "28%"},
            {"value": "Dance", "description": "Espetáculos, workshops, vídeos", "avg_success": "65%"},
            {"value": "Journalism", "description": "Reportagens, documentários jornalísticos", "avg_success": "24%"},
            {"value": "Crafts", "description": "Artesanato, DIY, kits", "avg_success": "27%"}
        ]
    }


@app.get("/info/countries", tags=["Information"])
async def get_countries():
    """Lista países suportados pelo modelo"""
    return {
        "total": 22,
        "main_countries": {
            "US": "Estados Unidos (70% dos projetos)",
            "GB": "Reino Unido (8% dos projetos)",
            "CA": "Canadá (4% dos projetos)",
            "AU": "Austrália (3% dos projetos)"
        },
        "all_countries": {
            "US": "Estados Unidos",
            "GB": "Reino Unido",
            "CA": "Canadá",
            "AU": "Austrália",
            "DE": "Alemanha",
            "FR": "França",
            "IT": "Itália",
            "ES": "Espanha",
            "NL": "Países Baixos",
            "SE": "Suécia",
            "NO": "Noruega",
            "DK": "Dinamarca",
            "IE": "Irlanda",
            "BE": "Bélgica",
            "CH": "Suíça",
            "AT": "Áustria",
            "NZ": "Nova Zelândia",
            "SG": "Singapura",
            "HK": "Hong Kong",
            "JP": "Japão",
            "MX": "México",
            "BR": "Brasil"
        }
    }
