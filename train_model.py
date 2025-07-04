# =====================================================
# PARTE 1: TREINAMENTO E TESTE DO MODELO
# =====================================================
# Arquivo: train_model.py

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

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


# =====================================================
# FUNÇÃO PRINCIPAL DE TREINAMENTO
# =====================================================

def train_kickstarter_model(data_path='ks-projects-201801.csv', save_model=True):
    """
    Treina o modelo completo do Kickstarter.
    
    Args:
        data_path: Caminho para o arquivo CSV
        save_model: Se deve salvar o modelo treinado
        
    Returns:
        Dicionário com modelo, preprocessador e métricas
    """
    
    print("="*80)
    print("TREINAMENTO DO MODELO KICKSTARTER")
    print("="*80)
    
    # 1. CARREGAR DADOS
    print("\n[1/8] Carregando dados...")
    df = pd.read_csv(data_path, encoding='latin-1')
    df.columns = df.columns.str.strip()
    print(f"✓ Dados carregados: {len(df):,} projetos")
    
    # 2. FILTRAR PROJETOS
    print("\n[2/8] Filtrando projetos finalizados...")
    df = df[df['state'].isin(['failed', 'successful'])]
    df['success'] = (df['state'] == 'successful').astype(int)
    print(f"✓ Projetos válidos: {len(df):,}")
    print(f"  - Sucessos: {df['success'].sum():,} ({df['success'].mean():.1%})")
    print(f"  - Falhas: {(1-df['success']).sum():,} ({(1-df['success']).mean():.1%})")
    
    # 3. DIVIDIR DADOS
    print("\n[3/8] Dividindo dados (80% treino, 20% teste)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['success'])
    print(f"✓ Treino: {len(train_df):,} projetos")
    print(f"✓ Teste: {len(test_df):,} projetos")
    
    # 4. CRIAR PREPROCESSADOR
    print("\n[4/8] Criando e ajustando preprocessador...")
    preprocessor = KickstarterPreprocessor()
    preprocessor.fit(train_df)
    print("✓ Preprocessador ajustado")
    
    # 5. TRANSFORMAR DADOS
    print("\n[5/8] Transformando dados...")
    X_train = preprocessor.transform(train_df)
    X_test = preprocessor.transform(test_df)
    y_train = train_df['success'].values
    y_test = test_df['success'].values
    print(f"✓ Dados transformados: {X_train.shape[1]} features")
    
    # 6. TREINAR MODELO
    print("\n[6/8] Treinando modelo Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=150,      # Número de árvores
        learning_rate=0.1,     # Taxa de aprendizado
        max_depth=5,           # Profundidade máxima das árvores
        min_samples_split=50,  # Mínimo de amostras para dividir nó
        min_samples_leaf=20,   # Mínimo de amostras em folha
        subsample=0.8,         # Fração de amostras por árvore
        random_state=42,
        verbose=1
    )
    model.fit(X_train, y_train)
    print("✓ Modelo treinado")
    
    # 7. AVALIAR MODELO
    print("\n[7/8] Avaliando modelo...")
    
    # Predições
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n✓ AUC-ROC: {auc_roc:.4f}")
    
    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-Validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Encontrar threshold ótimo
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    print(f"\nThreshold ótimo: {optimal_threshold:.3f}")
    
    # 8. SALVAR MODELO
    if save_model:
        print("\n[8/8] Salvando modelo...")
        model_data = {
            'model': model,
            'preprocessor': preprocessor,
            'optimal_threshold': optimal_threshold,
            'feature_names': preprocessor.features_selected,
            'version': '1.0',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'auc_roc': auc_roc,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'n_train': len(train_df),
                'n_test': len(test_df)
            }
        }
        
        filename = 'kickstarter_model_v1.pkl'
        joblib.dump(model_data, filename)
        print(f"✓ Modelo salvo como '{filename}'")
    
    print("\n" + "="*80)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*80)
    
    return model_data


# =====================================================
# FUNÇÃO DE TESTE DO MODELO
# =====================================================

def test_model(model_path='kickstarter_model_v1.pkl'):
    """
    Testa o modelo com alguns exemplos para verificar se está funcionando.
    """
    
    print("\n" + "="*80)
    print("TESTE DO MODELO")
    print("="*80)
    
    # Carregar modelo
    print("\nCarregando modelo...")
    model_data = joblib.load(model_path)
    
    # Criar predictor
    predictor = KickstarterPredictor(
        model=model_data['model'],
        preprocessor=model_data['preprocessor'],
        threshold=model_data['optimal_threshold']
    )
    
    print(f"✓ Modelo carregado (versão {model_data['version']})")
    print(f"✓ Treinado em: {model_data['training_date']}")
    print(f"✓ AUC-ROC: {model_data['metrics']['auc_roc']:.4f}")
    print(f"✓ Threshold: {model_data['optimal_threshold']:.3f}")
    
    # Exemplos de teste
    test_projects = [
        {
            'name': 'Revolutionary Smart Home Assistant with AI',
            'main_category': 'Technology',
            'country': 'US',
            'usd_goal_real': 50000,
            'launched': '2024-03-01',
            'deadline': '2024-03-31'
        },
        {
            'name': 'Eco-Friendly Water Bottle',
            'main_category': 'Design',
            'country': 'US',
            'usd_goal_real': 15000,
            'launched': '2024-04-01',
            'deadline': '2024-04-30'
        },
        {
            'name': 'Indie Game',
            'main_category': 'Games',
            'country': 'GB',
            'usd_goal_real': 5000,
            'launched': '2024-05-01',
            'deadline': '2024-05-30'
        },
        {
            'name': 'My First Album',
            'main_category': 'Music',
            'country': 'CA',
            'usd_goal_real': 2000,
            'launched': '2024-06-01',
            'deadline': '2024-06-20'
        }
    ]
    
    print("\n" + "-"*80)
    print("RESULTADOS DOS TESTES:")
    print("-"*80)
    
    for i, project in enumerate(test_projects, 1):
        print(f"\nTESTE {i}: {project['name']}")
        print(f"Categoria: {project['main_category']} | País: {project['country']} | Meta: ${project['usd_goal_real']:,}")
        
        # Fazer predição
        result = predictor.predict_single(project)
        
        print(f"\nResultado:")
        print(f"  - Probabilidade: {result['success_probability']:.1%}")
        print(f"  - Predição: {result['prediction']}")
        print(f"  - Confiança: {result['confidence']}")
        print(f"\nRecomendações:")
        for rec in result['recommendations']:
            print(f"  {rec}")
        
        print("-"*80)
    
    return predictor


# =====================================================
# EXECUTAR TREINAMENTO E TESTE
# =====================================================

if __name__ == "__main__":
    # Treinar modelo
    model_data = train_kickstarter_model(
        data_path='ks-projects-201801.csv',  # Ajuste o caminho se necessário
        save_model=True
    )
    
    # Testar modelo
    predictor = test_model('kickstarter_model_v1.pkl')
    
    print("\n✅ Modelo treinado e testado com sucesso!")
    print("\n📌 Próximo passo: Execute 'api.py' para criar a API")