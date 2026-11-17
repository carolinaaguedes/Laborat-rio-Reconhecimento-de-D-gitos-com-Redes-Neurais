"""Laboratório introdutório: redes neurais e reconhecimento de dígitos.

Sem APIs, downloads ou treinamento manual de neurônios.
É seguro executar novamente; gráficos serão substituídos.
"""

from pathlib import Path
import json
from time import perf_counter
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEMENTE = 42
SAIDA = Path('resultados')
SAIDA.mkdir(exist_ok=True)

# O conjunto de dados vem instalado com o scikit-learn.
dados = load_digits()
X, y = dados.data, dados.target
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=SEMENTE, stratify=y
)

# DESAFIO: após a primeira execução, altere uma arquitetura e compare.
CONFIGURACOES = {
    'MLP 16': (16,),
    'MLP 32': (32,),
}

resultados = []
modelos = {}
previsoes = {}

print(f'Dataset: {len(y)} imagens de 8x8 pixels, classes 0-9')
print(f'Treino: {len(y_treino)} | Teste: {len(y_teste)}')

for nome, arquitetura in CONFIGURACOES.items():
    pipeline = Pipeline([
        ('escala', StandardScaler()),
        ('rede', MLPClassifier(
            hidden_layer_sizes=arquitetura,
            activation='relu',
            solver='adam',
            max_iter=180,
            random_state=SEMENTE,
            early_stopping=True,
            n_iter_no_change=12,
        )),
    ])

    inicio = perf_counter()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always', ConvergenceWarning)
        pipeline.fit(X_treino, y_treino)
    tempo = perf_counter() - inicio
    pred = pipeline.predict(X_teste)

    acuracia = float(accuracy_score(y_teste, pred))
    f1 = float(f1_score(y_teste, pred, average='macro'))
    modelos[nome], previsoes[nome] = pipeline, pred
    resultados.append({
        'modelo': nome,
        'arquitetura': list(arquitetura),
        'acuracia': acuracia,
        'f1_macro': f1,
        'tempo_treinamento_seg': round(tempo, 4),
        'epocas': int(pipeline.named_steps['rede'].n_iter_),
        'aviso_convergencia': any(issubclass(a.category, ConvergenceWarning) for a in avisos),
    })
    print(f'{nome}: acurácia={acuracia:.3f}; F1={f1:.3f}; tempo={tempo:.2f}s')

# Comparação visual dos desempenhos.
nomes = [item['modelo'] for item in resultados]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(nomes, [100 * item['acuracia'] for item in resultados])
ax.set_ylim(0, 100)
ax.set_ylabel('Acurácia no teste (%)')
ax.set_title('Comparação das configurações')
fig.tight_layout()
fig.savefig(SAIDA / 'comparacao.png', dpi=150)
plt.close(fig)

# Mostrar matriz de confusão do modelo de melhor acurácia.
melhor = max(resultados, key=lambda r: r['acuracia'])['modelo']
pred = previsoes[melhor]
matriz = confusion_matrix(y_teste, pred, labels=np.arange(10))
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matriz, cmap='Blues')
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel('Previsto'); ax.set_ylabel('Verdadeiro')
ax.set_title(f'Matriz de confusão — {melhor}')
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(SAIDA / 'matriz_confusao.png', dpi=150)
plt.close(fig)

# Ver casos concretos de erros, não apenas a acurácia agregada.
erros = np.where(pred != y_teste)[0]
fig, eixos = plt.subplots(2, 4, figsize=(10, 5))
for ax, indice in zip(eixos.flat, erros[:8]):
    ax.imshow(X_teste[indice].reshape(8, 8), cmap='gray_r')
    ax.set_title(f'Real: {y_teste[indice]} | Pred.: {pred[indice]}', fontsize=10)
    ax.axis('off')
for ax in list(eixos.flat)[len(erros[:8]):]:
    ax.axis('off')
fig.suptitle(f'Erros do modelo {melhor} (primeiros {min(8,len(erros))})')
fig.tight_layout()
fig.savefig(SAIDA / 'erros.png', dpi=150)
plt.close(fig)

resumo = {
    'amostras': int(len(y)),
    'treino': int(len(y_treino)),
    'teste': int(len(y_teste)),
    'melhor_no_teste_nesta_execucao': melhor,
    'resultados': resultados,
    'erros_exibidos': [
        {'indice_teste': int(i), 'real': int(y_teste[i]), 'previsto': int(pred[i])}
        for i in erros[:8]
    ],
}
(SAIDA / 'resumo.json').write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False), encoding='utf-8'
)
print('\nGráficos e resumo foram salvos em resultados/')
print('Abra erros.png, compare os dígitos e preencha parecer.md.')
