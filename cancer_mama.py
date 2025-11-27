import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Definir o arquivo de dados
NOME_ARQUIVO = 'breast-cancer.csv'
SEED = 42 # Para resultados reproduzíveis

# ====================================================================
# 1. CARREGAR, PRÉ-PROCESSAR E DIVIDIR OS DADOS
# ====================================================================
print("--- 1. Preparação e Pré-Processamento dos Dados ---")

try:
    df = pd.read_csv(NOME_ARQUIVO)
except FileNotFoundError:
    print(f"ERRO: Arquivo '{NOME_ARQUIVO}' não encontrado. Certifique-se de que está no mesmo diretório.")
    exit()

# --- TRATAMENTO DE VALORES AUSENTES ('?') ---
for column in df.columns:
    if '?' in df[column].values:
        mode_val = df[column].mode()[0]
        df[column] = df[column].replace('?', mode_val)

dados_atributos = df.drop(columns=['Class'])
dados_classes = df['Class']

# --- CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS (One-Hot Encoding) ---
dados_atributos_encoded = pd.get_dummies(dados_atributos, drop_first=True)
colunas_atributos = dados_atributos_encoded.columns # Salva colunas para classificação de nova instância

# 2. DIVISÃO DOS DADOS
atributos_train, atributos_test, classes_train, classes_test = train_test_split(
    dados_atributos_encoded, dados_classes, test_size=0.3, random_state=SEED, stratify=dados_classes
)


# ====================================================================
# 2. TREINAMENTO DO MODELO (DecisionTree)
# ====================================================================
print("\n--- 2. Treinamento do Modelo Decision Tree ---")
tree = DecisionTreeClassifier(random_state=SEED)
modelo_dt = tree.fit(atributos_train, classes_train)
print("Modelo DecisionTreeClassifier treinado com sucesso!")

# 3. PREDIÇÃO NOS DADOS DE TESTE
classes_preditas = modelo_dt.predict(atributos_test)


# ====================================================================
# 3. COMPARAÇÃO ENTRE CLASSE REAL E PREVISTA
# ====================================================================
print("\n--- 3. Comparação de Classe Real (Teste) vs. Classe Prevista ---")

# Criar um DataFrame para facilitar a visualização e indexação
df_comparacao = pd.DataFrame({
    'Classe Real': classes_test,
    'Classe Prevista': classes_preditas
}).reset_index(drop=True)

# Imprimir as 10 primeiras e as 10 últimas amostras para exemplo
print("\nExemplo das Primeiras 10 Amostras:")
print(df_comparacao.head(10).to_markdown(index=False))

print("\nExemplo das Últimas 10 Amostras:")
print(df_comparacao.tail(10).to_markdown(index=False))

# Opcional: Imprimir a contagem total de acertos e erros (Apenas para demonstração)
acertos = (df_comparacao['Classe Real'] == df_comparacao['Classe Prevista']).sum()
total = len(df_comparacao)
erros = total - acertos
print(f"\nTotal de Amostras Testadas: {total}")
print(f"Total de Classificações Corretas: {acertos}")
print(f"Total de Classificações Incorretas: {erros}")


# ====================================================================
# 4. AVALIAÇÃO: ACURÁCIA E MATRIZ DE CONFUSÃO
# ====================================================================
print("\n" + "="*50)
print("--- 4. Demonstração da Acurácia e Matriz de Confusão ---")

# --- ACURÁCIA GLOBAL ---
acuracia_global = metrics.accuracy_score(classes_test, classes_preditas)
print(f'**Acurácia Global do Modelo (Score): {acuracia_global:.4f}**\n')

# --- MATRIZ DE CONFUSÃO (Contingência) ---
print("Matriz de Confusão (Dados Brutos):")
matriz = confusion_matrix(classes_test, classes_preditas, labels=modelo_dt.classes_)
df_matriz = pd.DataFrame(matriz, index=modelo_dt.classes_, columns=modelo_dt.classes_)
print(df_matriz)

# Visualização Gráfica da Matriz de Confusão
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=modelo_dt.classes_)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Matriz de Confusão Decision Tree')
plt.show()


# ====================================================================
# 5. CLASSIFICAR EXEMPLOS DE NOVAS INSTÂNCIAS
# ====================================================================
print("\n" + "="*50)
print("--- 5. Classificação de Nova Instância ---")

# Exemplo de um novo paciente (Instância de Teste)
nova_instancia_data = {
    'age': '50-59', 'menopause': 'ge40', 'tumor-size': '15-19', 'inv-nodes': '0-2',
    'node-caps': 'no', 'deg-malig': 2, 'breast': 'right', 'breast-quad': 'central',
    'irradiat': 'no'
}

nova_instancia_df = pd.DataFrame([nova_instancia_data])
nova_instancia_encoded = pd.get_dummies(nova_instancia_df)

# Reindexar e preencher com 0s para igualar a estrutura do treino
nova_instancia_final = nova_instancia_encoded.reindex(columns=colunas_atributos, fill_value=0)

# Fazer a predição
previsao = modelo_dt.predict(nova_instancia_final)

print("\nCaracterísticas da Nova Instância:")
for k, v in nova_instancia_data.items():
    print(f"- {k}: {v}")

print(f"\nO modelo Decision Tree prevê para esta nova instância a classe: **{previsao[0]}**")
print("=" * 50)