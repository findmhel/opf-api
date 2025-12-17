from pydantic import BaseModel
from typing import Optional

class UserData(BaseModel):
    """
    Schema para dados de entrada do modelo de predição de adesão ao Open Finance.
    
    Campos categóricos devem corresponder aos valores usados no treinamento.
    Campos numéricos são binários (0 ou 1) indicando presença/ausência ou contagem.
    """
    
    # Variáveis categóricas
    Faixa_etaria: str  # Ex: "25-34", "35-44", etc.
    Estado: str  # Ex: "SP", "RJ", etc.
    Sexo: str  # Ex: "M", "F"
    Ocupacao: str  # Ex: "CLT", "Autônomo", etc.
    Escolaridade: str  # Ex: "Superior", "Médio", etc.
    Gp_renda: str  # Ex: "Média", "Alta", "Baixa"
    Tipo_da_conta: str  # Ex: "Digital", "Tradicional"
    Gp_score_de_credito: str  # Ex: "Alto", "Médio", "Baixo"
    Gp_limite_do_cartao: str  # Ex: "Alto", "Médio", "Baixo"
    Tempo_conta_atv: str  # Ex: "2-5 anos", "< 2 anos", etc.
    
    # Variáveis numéricas (binárias ou contagem)
    Outros_bancos: int  # 0 ou 1
    Emprestimo: int  # 0 ou 1
    Financiamento: int  # 0 ou 1
    Cartao_de_credito: int  # 0 ou 1
    Usa_cheque: int  # 0 ou 1
    Atrasa_pag: int  # 0 ou 1
    Investimentos: int  # 0 ou 1
    Usa_pix: int  # 0 ou 1
    Usa_eBanking: int  # 0 ou 1
    Usa_app_banco: int  # 0 ou 1
