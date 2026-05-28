# Dados

Arquivos grandes (parquet, `.dbc`) não devem ser versionados sem necessidade; use `.gitignore` e mantenha aqui apenas o que for imprescindível para reprodução local.

## Dataset principal (definido)

- **Fonte:** SINASC (Sistema de Informações sobre Nascidos Vivos) / DATASUS.
- **Acesso no projeto:** [PySUS](https://pypi.org/project/pysus/) — ver `notebooks/01_eda.ipynb` para download (ex.: DN em **Minas Gerais**, anos **2020–2022**) e gravação dos parquet em `data/`.
- **Documentação de variáveis:** dicionários oficiais do DATASUS / OpenDataSUS para interpretação de códigos e sentinelas.

Atualize este arquivo se o grupo fixar outro recorte (UF, anos ou subconjunto de colunas) para o relatório e para o pipeline final.
