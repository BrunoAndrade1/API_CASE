import os
import gdown
import sys

def download_model():
    """Baixa o modelo do Google Drive se não existir"""
    model_path = 'kickstarter_model_v1.pkl'
    
    if os.path.exists(model_path):
        print("✅ Modelo já existe!")
        return True
    
    # Pegar FILE_ID da variável de ambiente ou usar default
    file_id = os.getenv('MODEL_FILE_ID', 'COLOQUE_SEU_FILE_ID_AQUI')
    
    if file_id == 'COLOQUE_SEU_FILE_ID_AQUI':
        print("❌ ERRO: Configure o MODEL_FILE_ID!")
        print("1. Faça upload do modelo para o Google Drive")
        print("2. Compartilhe como 'Qualquer pessoa com o link'")
        print("3. Copie o ID do arquivo da URL")
        print("4. Configure MODEL_FILE_ID no Render")
        return False
    
    try:
        print(f"📥 Baixando modelo do Google Drive...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
        print("✅ Modelo baixado com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao baixar modelo: {e}")
        return False

if __name__ == "__main__":
    if not download_model():
        sys.exit(1)
