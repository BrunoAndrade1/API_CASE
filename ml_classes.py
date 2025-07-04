import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =====================================================
# CLASSES DO MODELO
# =====================================================

class KickstarterPreprocessor:
    """
    Classe responsável por processar os dados do Kickstarter.
    Transforma dados brutos em features prontas para o modelo.
    """
    
    def __init__(self):
        # Dicionário para guardar os encoders de cada variável categórica
        self.label_encoders = {}
        
        # Scaler para normalizar as features numéricas
        self.scaler = StandardScaler()
        
        # Estatísticas que serão calculadas durante o fit
        self.category_stats = None
        self.country_stats = None
        
        # Lista de features que o modelo usará
        self.features_selected = [
            'cat_success_rate',      # Taxa de sucesso da categoria
            'usd_goal_real',         # Meta em USD
            'campaign_days',         # Duração da campanha
            'goal_magnitude',        # Log da meta (captura escala)
            'cat_mean_goal',         # Meta média da categoria
            'name_word_count',       # Palavras no título
            'cat_median_goal',       # Meta mediana da categoria
            'goal_per_day',          # Meta dividida por dias
            'country_success_rate',  # Taxa de sucesso do país
            'launch_year',           # Ano de lançamento
            'main_category',         # Categoria (encoded)
            'name_length',           # Comprimento do título
            'goal_category_ratio',   # Razão meta/mediana categoria
            'country',               # País (encoded)
            'goal_rounded'           # Se a meta é "redonda" (ex: 5000)
        ]
    
    def create_features(self, df):
        """
        Cria features básicas a partir dos dados brutos.
        Esta função é chamada tanto no fit quanto no transform.
        """
        df = df.copy()
        
        # 1. Converter datas para datetime
        df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
        df['launched'] = pd.to_datetime(df['launched'], errors='coerce')
        
        # 2. Calcular duração da campanha em dias
        df['campaign_days'] = (df['deadline'] - df['launched']).dt.days
        
        # 3. Extrair ano de lançamento
        df['launch_year'] = df['launched'].dt.year
        
        # 4. Validar campaign_days (mínimo 1, máximo 365)
        df['campaign_days'] = df['campaign_days'].clip(lower=1, upper=365)
        
        # 5. Features do título/nome do projeto
        df['name_length'] = df['name'].fillna('').str.len()
        df['name_word_count'] = df['name'].fillna('').str.split().str.len()
        
        # 6. Limitar meta máxima para evitar outliers extremos
        df['usd_goal_real'] = df['usd_goal_real'].clip(upper=1e8)  # Max 100 milhões
        
        # 7. Magnitude logarítmica da meta (captura ordem de grandeza)
        df['goal_magnitude'] = np.log10(df['usd_goal_real'].clip(lower=1) + 1)
        
        # 8. Se a meta é um número "redondo" (termina em 000)
        df['goal_rounded'] = (df['usd_goal_real'] % 1000 == 0).astype(int)
        
        return df
    
    def fit(self, df):
        """
        Ajusta o preprocessador com os dados de treino.
        Calcula estatísticas que serão usadas para transformar dados futuros.
        """
        # Criar features básicas
        df = self.create_features(df)
        
        # Calcular estatísticas por categoria
        print("Calculando estatísticas por categoria...")
        self.category_stats = df.groupby('main_category').agg({
            'state': lambda x: (x == 'successful').mean(),  # Taxa de sucesso
            'usd_goal_real': ['mean', 'median']            # Meta média e mediana
        }).round(3)
        self.category_stats.columns = ['cat_success_rate', 'cat_mean_goal', 'cat_median_goal']
        
        # Calcular estatísticas por país
        print("Calculando estatísticas por país...")
        self.country_stats = df.groupby('country').agg({
            'state': lambda x: (x == 'successful').mean()   # Taxa de sucesso
        }).round(3)
        self.country_stats.columns = ['country_success_rate']
        
        # Aplicar estatísticas ao dataframe
        df = df.merge(self.category_stats, left_on='main_category', right_index=True, how='left')
        df = df.merge(self.country_stats, left_on='country', right_index=True, how='left')
        
        # Criar features derivadas
        df['goal_per_day'] = df['usd_goal_real'] / df['campaign_days'].replace(0, 1)
        df['goal_category_ratio'] = df['usd_goal_real'] / df['cat_median_goal'].replace(0, 1)
        
        # Tratar valores infinitos e NaN
        df['goal_per_day'] = df['goal_per_day'].replace([np.inf, -np.inf], 0).fillna(0)
        df['goal_category_ratio'] = df['goal_category_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
        
        # Criar e ajustar label encoders
        print("Criando encoders para variáveis categóricas...")
        self.label_encoders['main_category'] = LabelEncoder()
        self.label_encoders['country'] = LabelEncoder()
        
        df['main_category'] = self.label_encoders['main_category'].fit_transform(df['main_category'])
        df['country'] = self.label_encoders['country'].fit_transform(df['country'])
        
        # Ajustar scaler com as features selecionadas
        print("Ajustando normalizador...")
        X = df[self.features_selected]
        self.scaler.fit(X)
        
        return self
    
    def transform(self, df):
        """
        Transforma novos dados usando as estatísticas calculadas no fit.
        Esta função é usada tanto para dados de teste quanto para produção.
        """
        # Criar features básicas
        df = self.create_features(df)
        
        # Aplicar estatísticas (com valores padrão para categorias/países novos)
        df = df.merge(self.category_stats, left_on='main_category', right_index=True, how='left')
        df = df.merge(self.country_stats, left_on='country', right_index=True, how='left')
        
        # Preencher valores faltantes com valores padrão
        # (para categorias/países que não existiam no treino)
        df['cat_success_rate'].fillna(0.35, inplace=True)      # Taxa média geral
        df['cat_mean_goal'].fillna(10000, inplace=True)        # Meta média geral
        df['cat_median_goal'].fillna(5000, inplace=True)       # Meta mediana geral
        df['country_success_rate'].fillna(0.35, inplace=True)  # Taxa média geral
        
        # Criar features derivadas
        df['goal_per_day'] = df['usd_goal_real'] / df['campaign_days'].replace(0, 1)
        df['goal_category_ratio'] = df['usd_goal_real'] / df['cat_median_goal'].replace(0, 1)
        
        # Tratar valores infinitos e NaN
        df['goal_per_day'] = df['goal_per_day'].replace([np.inf, -np.inf], 0).fillna(0)
        df['goal_category_ratio'] = df['goal_category_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
        
        # Aplicar encoders (com tratamento para valores novos)
        for col, encoder in self.label_encoders.items():
            known_values = set(encoder.classes_)
            # Se o valor não foi visto no treino, usar o primeiro valor conhecido
            df[col] = df[col].apply(lambda x: x if x in known_values else list(known_values)[0])
            df[col] = encoder.transform(df[col])
        
        # Selecionar e normalizar features
        X = df[self.features_selected]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled


class KickstarterPredictor:
    """
    Classe para fazer predições e gerar recomendações.
    Encapsula o modelo e o preprocessador.
    """
    
    def __init__(self, model, preprocessor, threshold=0.5):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold
    
    def predict_single(self, project_data):
        """
        Faz predição para um único projeto.
        
        Args:
            project_data: Dicionário com os dados do projeto
            
        Returns:
            Dicionário com predição, probabilidade e recomendações
        """
        # Converter para DataFrame
        df = pd.DataFrame([project_data])
        
        # Processar dados
        X = self.preprocessor.transform(df)
        
        # Fazer predição
        proba = self.model.predict_proba(X)[0, 1]
        prediction = int(proba >= self.threshold)
        
        # Gerar recomendações personalizadas
        recommendations = self._generate_recommendations(project_data, proba)
        
        return {
            'success_probability': float(proba),
            'prediction': 'Sucesso' if prediction else 'Falha',
            'confidence': self._calculate_confidence(proba),
            'recommendations': recommendations,
            'threshold_used': self.threshold
        }
    
    def _calculate_confidence(self, proba):
        """Calcula o nível de confiança baseado na distância do threshold"""
        distance = abs(proba - self.threshold)
        if distance > 0.3:
            return 'Alta'
        elif distance > 0.15:
            return 'Média'
        else:
            return 'Baixa'
    
    def _generate_recommendations(self, project_data, proba):
        """Gera recomendações baseadas nos dados do projeto"""
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
            # Calcular se não foi fornecido
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
        
        # Recomendação geral baseada na probabilidade
        if proba < 0.3:
            recommendations.append("🔴 Risco alto de falha. Revise estratégia completa.")
        elif proba < 0.5:
            recommendations.append("🟡 Chances moderadas. Pequenos ajustes podem fazer diferença.")
        elif proba < 0.7:
            recommendations.append("🟢 Boas chances de sucesso. Mantenha execução forte.")
        else:
            recommendations.append("🌟 Excelentes chances! Foque na execução.")
        
        return recommendations
