"""
API FastAPI para o modelo Kickstarter Success Predictor

Para executar:
1. Certifique-se que o arquivo 'kickstarter_model_v1.pkl' existe
2. Instale as dependências: pip install fastapi uvicorn pandas scikit-learn joblib
3. Execute: python api.py
4. Acesse: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =====================================================
# CLASSES NECESSÁRIAS PARA O MODELO
# =====================================================

class KickstarterPreprocessor:
    """
    Classe para processar dados do Kickstarter de forma consistente.
    Esta classe precisa ser idêntica à usada no treinamento.
    """
    
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
        """Cria todas as features necessárias"""
        df = df.copy()
        
        # Garantir tipos corretos
        df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
        df['launched'] = pd.to_datetime(df['launched'], errors='coerce')
        
        # Features básicas
        df['campaign_days'] = (df['deadline'] - df['launched']).dt.days
        df['launch_year'] = df['launched'].dt.year
        
        # Validar campaign_days
        df['campaign_days'] = df['campaign_days'].clip(lower=1, upper=365)
        
        # Features de texto
        df['name_length'] = df['name'].fillna('').str.len()
        df['name_word_count'] = df['name'].fillna('').str.split().str.len()
        
        # Features de meta
        df['usd_goal_real'] = df['usd_goal_real'].clip(upper=1e8)
        df['goal_magnitude'] = np.log10(df['usd_goal_real'].clip(lower=1) + 1)
        df['goal_rounded'] = (df['usd_goal_real'] % 1000 == 0).astype(int)
        
        return df
    
    def transform(self, df):
        """Transforma novos dados usando os transformadores ajustados"""
        df = self.create_features(df)
        
        # Aplicar estatísticas
        df = df.merge(self.category_stats, left_on='main_category', right_index=True, how='left')
        df = df.merge(self.country_stats, left_on='country', right_index=True, how='left')
        
        # Preencher valores faltantes
        df['cat_success_rate'].fillna(0.35, inplace=True)
        df['cat_mean_goal'].fillna(10000, inplace=True)
        df['cat_median_goal'].fillna(5000, inplace=True)
        df['country_success_rate'].fillna(0.35, inplace=True)
        
        # Features derivadas
        df['goal_per_day'] = df['usd_goal_real'] / df['campaign_days'].replace(0, 1)
        df['goal_category_ratio'] = df['usd_goal_real'] / df['cat_median_goal'].replace(0, 1)
        
        # Tratar infinitos
        df['goal_per_day'] = df['goal_per_day'].replace([np.inf, -np.inf], 0).fillna(0)
        df['goal_category_ratio'] = df['goal_category_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
        
        # Aplicar encoders
        for col, encoder in self.label_encoders.items():
            known_values = set(encoder.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_values else list(known_values)[0])
            df[col] = encoder.transform(df[col])
        
        # Selecionar e escalar
        X = df[self.features_selected]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled


class KickstarterPredictor:
    """Classe para fazer predições e gerar recomendações"""
    
    def __init__(self, model, preprocessor, threshold=0.5):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold
    
    def predict_single(self, project_data):
        """Faz predição para um único projeto"""
        # Converter para DataFrame
        df = pd.DataFrame([project_data])
        
        # Processar
        X = self.preprocessor.transform(df)
        
        # Predizer
        proba = self.model.predict_proba(X)[0, 1]
        prediction = int(proba >= self.threshold)
        
        # Gerar recomendações
        recommendations = self._generate_recommendations(project_data, proba)
        
        return {
            'success_probability': float(proba),
            'prediction': 'Sucesso' if prediction else 'Falha',
            'confidence': self._calculate_confidence(proba),
            'recommendations': recommendations,
            'threshold_used': self.threshold
        }
    
    def _calculate_confidence(self, proba):
        """Calcula nível de confiança da predição"""
        distance = abs(proba - self.threshold)
        if distance > 0.3:
            return 'Alta'
        elif distance > 0.15:
            return 'Média'
        else:
            return 'Baixa'
    
    def _generate_recommendations(self, project_data, proba):
        """Gera recomendações personalizadas"""
        recommendations = []
        
        # Análise da meta
        goal = project_data.get('usd_goal_real', 0)
        if goal > 50000:
            recommendations.append("⚠️ Meta muito alta. Considere reduzir para aumentar chances.")
        elif goal < 1000:
            recommendations.append("✅ Meta modesta, boa estratégia para primeira campanha.")
        else:
            recommendations.append("✅ Meta dentro da faixa recomendada.")
        
        # Análise da duração
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
        
        # Análise do título
        name_words = len(project_data.get('name', '').split())
        if name_words < 3:
            recommendations.append("💡 Título muito curto. Seja mais descritivo.")
        elif name_words > 10:
            recommendations.append("💡 Título muito longo. Seja mais conciso.")
        
        # Recomendação geral
        if proba < 0.3:
            recommendations.append("🔴 Risco alto de falha. Revise estratégia completa.")
        elif proba < 0.5:
            recommendations.append("🟡 Chances moderadas. Pequenos ajustes podem fazer diferença.")
        elif proba < 0.7:
            recommendations.append("🟢 Boas chances de sucesso. Mantenha execução forte.")
        else:
            recommendations.append("🌟 Excelentes chances! Foque na execução.")
        
        return recommendations


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
# CARREGAR MODELO
# =====================================================

MODEL_PATH = 'kickstarter_model_v1.pkl'
model_data = None
predictor = None

def load_model():
    """Carrega o modelo do disco"""
    global model_data, predictor
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo não encontrado em '{MODEL_PATH}'. "
            "Execute o script de treinamento primeiro."
        )
    
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

# Tentar carregar modelo ao iniciar
try:
    load_model()
except Exception as e:
    print(f"⚠️ Erro ao carregar modelo: {e}")

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
        "endpoints": {
            "documentação_interativa": "/docs",
            "documentação_alternativa": "/redoc",
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
    """Verifica se a API está funcionando"""
    return {
        "status": "healthy" if predictor else "unhealthy",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_project(project: ProjectInput):
    """
    Faz predição para um único projeto Kickstarter.
    
    Retorna:
    - Probabilidade de sucesso (0-100%)
    - Predição final (Sucesso/Falha)
    - Nível de confiança
    - Recomendações personalizadas
    """
    
    if not predictor:
        raise HTTPException(
            status_code=503,
            detail="Modelo não está carregado. Reinicie a API."
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
    
    Útil para analisar portfólios ou comparar diferentes configurações.
    """
    
    if not predictor:
        raise HTTPException(
            status_code=503,
            detail="Modelo não está carregado."
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


@app.post("/reload-model", tags=["Admin"])
async def reload_model():
    """Recarrega o modelo do disco"""
    
    try:
        load_model()
        return {
            "status": "success",
            "message": "Modelo recarregado com sucesso",
            "model_version": model_data['version'],
            "training_date": model_data['training_date']
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao recarregar modelo: {str(e)}"
        )


@app.get("/example/curl", tags=["Examples"])
async def example_curl():
    """Exemplo de uso com cURL"""
    
    return {
        "description": "Exemplo de comando cURL para fazer uma predição",
        "command": """curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "name": "Revolutionary Smart Watch with AI",
       "main_category": "Technology",
       "country": "US",
       "usd_goal_real": 25000,
       "launched": "2024-03-01",
       "deadline": "2024-03-31"
     }'"""
    }


@app.get("/example/python", tags=["Examples"])
async def example_python():
    """Exemplo de uso com Python"""
    
    return {
        "description": "Exemplo de código Python para usar a API",
        "code": """import requests

# URL da API
url = "http://localhost:8000/predict"

# Dados do projeto
project = {
    "name": "Revolutionary Smart Watch with AI",
    "main_category": "Technology",
    "country": "US",
    "usd_goal_real": 25000,
    "launched": "2024-03-01",
    "deadline": "2024-03-31"
}

# Fazer requisição
response = requests.post(url, json=project)

# Processar resposta
if response.status_code == 200:
    result = response.json()
    print(f"Probabilidade de sucesso: {result['success_probability']:.1%}")
    print(f"Predição: {result['prediction']}")
    print(f"Confiança: {result['confidence']}")
    print("\\nRecomendações:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
else:
    print(f"Erro: {response.status_code}")
    print(response.json())"""
    }


@app.get("/example/javascript", tags=["Examples"])
async def example_javascript():
    """Exemplo de uso com JavaScript"""
    
    return {
        "description": "Exemplo de código JavaScript para usar a API",
        "code": """// Função para fazer predição
async function predictKickstarterProject() {
    const project = {
        name: "Revolutionary Smart Watch with AI",
        main_category: "Technology",
        country: "US",
        usd_goal_real: 25000,
        launched: "2024-03-01",
        deadline: "2024-03-31"
    };
    
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(project)
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log(`Probabilidade: ${(result.success_probability * 100).toFixed(1)}%`);
            console.log(`Predição: ${result.prediction}`);
            console.log(`Confiança: ${result.confidence}`);
            console.log('Recomendações:');
            result.recommendations.forEach(rec => console.log(`  - ${rec}`));
        } else {
            console.error('Erro:', await response.json());
        }
    } catch (error) {
        console.error('Erro de conexão:', error);
    }
}

// Executar
predictKickstarterProject();"""
    }


# =====================================================
# EXECUTAR SERVIDOR
# =====================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 KICKSTARTER SUCCESS PREDICTOR API")
    print("="*60)
    print("\n📌 Documentação interativa: http://localhost:8000/docs")
    print("📌 Documentação alternativa: http://localhost:8000/redoc")
    print("📌 Testar predição: POST http://localhost:8000/predict")
    print("\n✨ Dica: Use a documentação interativa para testar a API!")
    print("\nPressione CTRL+C para parar o servidor\n")
    
    # Configurações do servidor
    uvicorn.run(
        "api:app",  # Se o arquivo se chamar api.py
        host="127.0.0.1",  # Aceita conexões de qualquer IP
        port=8000,  # Porta padrão
        reload=True,  # Recarrega automaticamente ao salvar o arquivo
        log_level="info"  # Nível de log
    )