# Fascinator System - Architecture Diagrams

This document contains comprehensive Mermaid diagrams explaining the proposed Fascinator system architecture, data flows, and operational workflows.

---

## 1. HIGH-LEVEL SYSTEM ARCHITECTURE

```mermaid
graph TB
    subgraph External["External Sources"]
        WEB[("🌐 Web<br/>Domains")]
    end

    subgraph Ingestion["Ingestion Layer"]
        RC["Rust Crawler<br/>High Performance"]
        GC["Go Crawler<br/>Concurrent"]
        CF["Crawl Frontier<br/>Priority Queue"]

        RC --> CF
        GC --> CF
        WEB --> RC
        WEB --> GC
    end

    subgraph Storage["Storage Layer"]
        RAW[("📦 Raw Storage<br/>ZFS Dataset<br/>2TB NVMe")]
        PROC[("📄 Processed<br/>Parquet<br/>2TB NVMe")]
        ARCH[("🗄️ Archive<br/>Cold Storage")]
    end

    subgraph Processing["Processing Layer"]
        TIKA["Apache Tika<br/>Service"]
        NLP["NLP Pipeline<br/>spaCy/NLTK"]
        EMBED["Embedding Gen<br/>sentence-transformers"]
        TOPIC["Topic Modeling<br/>Gensim/LDA"]
    end

    subgraph Database["Database Layer"]
        PG[("🐘 PostgreSQL<br/>+pgvector")]
        REDIS[("⚡ Redis<br/>Cache + Queues")]
    end

    subgraph Intelligence["Intelligence Layer"]
        ICM["ICM Module<br/>Curiosity Engine"]
        SCORE["URL Scorer"]
        EVICT["Document Evictor"]
        CLUSTER["Clusterer<br/>scikit-learn"]
    end

    subgraph Orchestration["Orchestration Layer"]
        MQ["Message Queue<br/>RabbitMQ/Kafka"]
        WORK["Workflow Engine<br/>Airflow"]
        SCHED["Scheduler"]
    end

    subgraph API["API Layer"]
        REST["REST API<br/>FastAPI"]
        DASH["Admin Dashboard<br/>React"]
        CLI["CLI Tool"]
    end

    subgraph Monitoring["Monitoring Layer"]
        PROM["Prometheus"]
        GRAF["Grafana"]
        ELK["ELK Stack"]
        ALERT["Alertmanager"]
    end

    %% Flows
    CF --> RAW
    RAW --> TIKA
    TIKA --> NLP
    NLP --> EMBED
    NLP --> TOPIC
    TOPIC --> PG
    EMBED --> PG
    PROC --> PG

    PG --> ICM
    REDIS --> ICM
    ICM --> SCORE
    ICM --> EVICT
    ICM --> CLUSTER

    SCORE --> CF
    EVICT --> ARCH
    CLUSTER --> PG

    MQ --> RC
    MQ --> GC
    MQ --> TIKA
    MQ --> ICM

    WORK --> SCHED
    SCHED --> MQ

    REST --> PG
    REST --> REDIS
    DASH --> REST
    CLI --> REST

    PROM --> GRAF
    ELK --> ALERT

    classDef storage fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    classDef processing fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef intelligence fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef api fill:#e8f5e9,stroke:#388e3c,stroke-width:2px

    class RAW,PROC,ARCH,PG,REDIS storage
    class TIKA,NLP,EMBED,TOPIC processing
    class ICM,SCORE,EVICT,CLUSTER intelligence
    class REST,DASH,CLI api
```

---

## 2. DETAILED DATA FLOW PIPELINE

```mermaid
flowchart TD
    START([Start: URL Seed]) --> Q1{URL in<br/>DB?}
    Q1 -->|Yes| Q2{Recently<br/>Crawled?}
    Q1 -->|No| CRAWL
    Q2 -->|Yes| END1([Skip])
    Q2 -->|No| CRAWL

    CRAWL[Crawl URL] --> Q3{Success?}
    Q3 -->|No| LOG1[Log Error] --> MQ1[Requeue with<br/>Backoff]
    MQ1 --> END2([End])
    Q3 -->|Yes| SAVE1[Save Raw File<br/>to ZFS]

    SAVE1 --> DETECT[Detect File Type]
    DETECT --> TIKA[Apache Tika<br/>Extract Text]

    TIKA --> Q4{Success?}
    Q4 -->|No| LOG2[Log Error] --> ARCH1[Archive<br/>Failed File]
    ARCH1 --> END3([End])
    Q4 -->|Yes| CLEAN[Clean Text<br/>Remove Headers/Footers]

    CLEAN --> LANG[Detect Language]
    LANG --> Q5{Supported<br/>Language?}
    Q5 -->|No| ARCH2[Archive<br/>Unsupported]
    ARCH2 --> END4([End])
    Q5 -->|Yes| NLP

    NLP[NLP Processing<br/>Tokenize, NER, POS] --> EMBED[Generate<br/>Embeddings]
    NLP --> TOPIC[Topic<br/>Modeling]

    EMBED --> PGVEC[Store in<br/>pgvector]
    TOPIC --> PGMETA[Store Topic<br/>Metadata]

    PGVEC --> REDIS1[Cache Vector<br/>in Redis]
    PGMETA --> REDIS2[Cache Topics<br/>in Redis]

    REDIS1 --> ICM[ICM: Compute<br/>Curiosity Score]
    REDIS2 --> ICM

    ICM --> Q6{Score ><br/>Threshold?}
    Q6 -->|No| EVICT[Evict Document<br/>to Archive]
    EVICT --> DEL[Delete from<br/>Hot Storage]
    DEL --> END5([End])

    Q6 -->|Yes| KEEP[Keep Document<br/>in Hot Storage]
    KEEP --> EXTRACT[Extract URLs<br/>from Document]

    EXTRACT --> URLSCORE[Score Each URL<br/>for Curiosity]
    URLSCORE --> URLCACHE[Cache High-Score<br/>URLs in Redis]

    URLCACHE --> FRONTIER[Add to Crawl<br/>Frontier Queue]
    FRONTIER --> Q7{More<br/>URLs?}
    Q7 -->|Yes| START
    Q7 -->|No| END6([End])

    style START fill:#4caf50,stroke:#2e7d32,color:#fff
    style END1 fill:#9e9e9e,stroke:#616161,color:#fff
    style END2 fill:#9e9e9e,stroke:#616161,color:#fff
    style END3 fill:#9e9e9e,stroke:#616161,color:#fff
    style END4 fill:#9e9e9e,stroke:#616161,color:#fff
    style END5 fill:#9e9e9e,stroke:#616161,color:#fff
    style END6 fill:#4caf50,stroke:#2e7d32,color:#fff
    style ICM fill:#9c27b0,stroke:#6a1b9a,color:#fff
    style EVICT fill:#f44336,stroke:#c62828,color:#fff
```

---

## 3. ICM (INTRINSIC CURIOSITY MODULE) DETAILED WORKFLOW

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        DOC[New Document]
        DOCMETA[Document Metadata<br/>URL, Domain, Timestamp]
    end

    subgraph FeatureExtraction["Feature Extraction"]
        EMBED1[Embedding Generator<br/>sentence-transformers]
        META1[Metadata Encoder<br/>Domain, Type, Depth]
        DOC --> EMBED1
        DOCMETA --> META1
    end

    subgraph StateRepresentation["State Representation"]
        CONCAT1[Concatenate Features]
        EMBED1 --> CONCAT1
        META1 --> CONCAT1

        STATE[State Vector s_t<br/>Dimension: 768]
        CONCAT1 --> STATE
    end

    subgraph HistoricalContext["Historical Context"]
        PGQUERY[Query pgvector<br/>for Similar Docs]
        STATE --> PGQUERY

        NEIGHBORS[K-Nearest Neighbors<br/>Previously Seen]
        PGQUERY --> NEIGHBORS

        PREVSTATE[Previous State s_t-1]
        NEIGHBORS --> PREVSTATE
    end

    subgraph ICMCore["ICM Core Models"]
        direction TB

        subgraph InverseModel["Inverse Model"]
            INV_IN["Input: [s_t-1, s_t]"]
            INV_NET["Neural Network<br/>3 Dense Layers"]
            INV_OUT["Output: â_t<br/>Predicted Action"]

            INV_IN --> INV_NET --> INV_OUT
        end

        subgraph ForwardModel["Forward Model"]
            FWD_IN["Input: [s_t-1, a_t]"]
            FWD_NET["Neural Network<br/>3 Dense Layers"]
            FWD_OUT["Output: ŝ_t<br/>Predicted Next State"]

            FWD_IN --> FWD_NET --> FWD_OUT
        end

        STATE --> INV_IN
        PREVSTATE --> INV_IN
        PREVSTATE --> FWD_IN

        ACTION[Actual Action a_t<br/>URL followed, domain explored]
        ACTION --> FWD_IN
    end

    subgraph Reward["Intrinsic Reward Computation"]
        DIFF["Prediction Error<br/>||ŝ_t - s_t||²"]
        STATE --> DIFF
        FWD_OUT --> DIFF

        SCALE["Scale by η<br/>Learning Rate"]
        DIFF --> SCALE

        INTRINSIC["Intrinsic Reward r_i<br/>Novelty Score"]
        SCALE --> INTRINSIC
    end

    subgraph Scoring["Document & URL Scoring"]
        DOCSCORE["Document Score<br/>= r_i × importance"]
        INTRINSIC --> DOCSCORE

        THRESHOLD{Score ><br/>Threshold?}
        DOCSCORE --> THRESHOLD

        KEEP["✓ Keep Document<br/>High Novelty"]
        EVICT["✗ Evict Document<br/>Low Novelty"]

        THRESHOLD -->|Yes| KEEP
        THRESHOLD -->|No| EVICT
    end

    subgraph URLExtraction["URL Curiosity Scoring"]
        EXTRACT["Extract URLs<br/>from Document"]
        KEEP --> EXTRACT

        URLFEATURES["URL Features<br/>Domain, TLD, Path Depth"]
        EXTRACT --> URLFEATURES

        URLPREDICT["Predict Future<br/>Curiosity Score"]
        URLFEATURES --> URLPREDICT
        STATE --> URLPREDICT

        URLRANK["Rank URLs by<br/>Predicted Curiosity"]
        URLPREDICT --> URLRANK
    end

    subgraph Output["Output Actions"]
        CACHE["Cache High-Score<br/>URLs in Redis"]
        URLRANK --> CACHE

        QUEUE["Add to Crawl<br/>Priority Queue"]
        CACHE --> QUEUE

        ARCHIVE["Move Low-Score<br/>Docs to Archive"]
        EVICT --> ARCHIVE

        UPDATE["Update ICM<br/>Model Weights"]
        INTRINSIC --> UPDATE
    end

    subgraph Feedback["Feedback Loop"]
        RETRAIN["Periodic<br/>Retraining"]
        UPDATE --> RETRAIN

        METRICS["Track Metrics<br/>Exploration Rate<br/>Diversity Score"]
        RETRAIN --> METRICS

        ADJUST["Adjust<br/>Hyperparameters"]
        METRICS --> ADJUST

        ADJUST -.->|Update| ICMCore
    end

    style ICMCore fill:#e1bee7,stroke:#8e24aa,stroke-width:3px
    style InverseModel fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style ForwardModel fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style INTRINSIC fill:#c5e1a5,stroke:#558b2f,stroke-width:3px
    style KEEP fill:#81c784,stroke:#388e3c,stroke-width:2px
    style EVICT fill:#e57373,stroke:#c62828,stroke-width:2px
```

---

## 4. MESSAGE QUEUE ARCHITECTURE

```mermaid
graph TB
    subgraph Producers["Message Producers"]
        CRON["Cron Jobs"]
        API1["REST API"]
        ICM1["ICM Module"]
        CRAWLER1["Crawlers"]
    end

    subgraph RabbitMQ["RabbitMQ / Kafka"]
        EX1["Exchange:<br/>crawl.tasks"]
        EX2["Exchange:<br/>process.tasks"]
        EX3["Exchange:<br/>analysis.tasks"]
        EX4["Exchange:<br/>eviction.tasks"]

        Q1["Queue:<br/>crawl.high_priority"]
        Q2["Queue:<br/>crawl.normal"]
        Q3["Queue:<br/>crawl.low_priority"]

        Q4["Queue:<br/>tika.processing"]
        Q5["Queue:<br/>nlp.processing"]

        Q6["Queue:<br/>topic_modeling"]
        Q7["Queue:<br/>embedding_gen"]
        Q8["Queue:<br/>icm_scoring"]

        Q9["Queue:<br/>eviction"]
        Q10["Queue:<br/>archival"]

        DLQ1["DLQ:<br/>crawl.failed"]
        DLQ2["DLQ:<br/>process.failed"]

        EX1 --> Q1
        EX1 --> Q2
        EX1 --> Q3

        EX2 --> Q4
        EX2 --> Q5

        EX3 --> Q6
        EX3 --> Q7
        EX3 --> Q8

        EX4 --> Q9
        EX4 --> Q10

        Q1 -.->|Failed| DLQ1
        Q2 -.->|Failed| DLQ1
        Q3 -.->|Failed| DLQ1

        Q4 -.->|Failed| DLQ2
        Q5 -.->|Failed| DLQ2
    end

    subgraph Consumers["Message Consumers"]
        CW1["Crawler Worker 1"]
        CW2["Crawler Worker 2"]
        CW3["Crawler Worker N"]

        PW1["Processing Worker 1"]
        PW2["Processing Worker 2"]

        AW1["Analysis Worker 1"]
        AW2["Analysis Worker 2"]

        EW1["Eviction Worker"]
    end

    subgraph Monitoring["Queue Monitoring"]
        MON["Queue Monitor<br/>Prometheus Exporter"]
        ALERT1["Alerting:<br/>Queue Backup"]
        ALERT2["Alerting:<br/>Consumer Lag"]
    end

    %% Producer to Exchange
    CRON --> EX1
    API1 --> EX1
    ICM1 --> EX1
    ICM1 --> EX4
    CRAWLER1 --> EX2
    PW1 --> EX3
    PW2 --> EX3

    %% Queue to Consumer
    Q1 --> CW1
    Q2 --> CW2
    Q3 --> CW3

    Q4 --> PW1
    Q5 --> PW2

    Q6 --> AW1
    Q7 --> AW1
    Q8 --> AW2

    Q9 --> EW1
    Q10 --> EW1

    %% Monitoring
    RabbitMQ --> MON
    MON --> ALERT1
    MON --> ALERT2

    classDef producer fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef queue fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef consumer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef dlq fill:#ffebee,stroke:#c62828,stroke-width:2px

    class CRON,API1,ICM1,CRAWLER1 producer
    class Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10 queue
    class CW1,CW2,CW3,PW1,PW2,AW1,AW2,EW1 consumer
    class DLQ1,DLQ2 dlq
```

---

## 5. DATABASE SCHEMA & RELATIONSHIPS

```mermaid
erDiagram
    DOCUMENTS ||--o{ DOCUMENT_VECTORS : contains
    DOCUMENTS ||--o{ DOCUMENT_TOPICS : has
    DOCUMENTS ||--|| CRAWL_METADATA : from
    DOCUMENTS ||--o{ NOVELTY_SCORES : scored_as
    DOCUMENTS ||--o{ CLUSTER_ASSIGNMENTS : belongs_to

    URLS ||--o{ CRAWL_QUEUE : queued_in
    URLS ||--o{ URL_CURIOSITY_SCORES : has
    URLS ||--|| CRAWL_METADATA : produces

    TOPICS ||--o{ DOCUMENT_TOPICS : assigned_to
    TOPICS ||--o{ TOPIC_EVOLUTION : evolves_as

    CLUSTERS ||--o{ CLUSTER_ASSIGNMENTS : contains

    ICM_MODELS ||--o{ NOVELTY_SCORES : generates
    ICM_MODELS ||--o{ URL_CURIOSITY_SCORES : predicts
    ICM_MODELS ||--o{ MODEL_CHECKPOINTS : versioned_as

    DOCUMENTS {
        uuid id PK
        text content
        text raw_path
        text processed_path
        timestamp crawled_at
        timestamp processed_at
        string language
        string content_type
        jsonb metadata
        boolean is_archived
    }

    DOCUMENT_VECTORS {
        uuid id PK
        uuid document_id FK
        vector embedding "pgvector(768)"
        string model_version
        timestamp created_at
    }

    DOCUMENT_TOPICS {
        uuid id PK
        uuid document_id FK
        uuid topic_id FK
        float probability
        timestamp assigned_at
    }

    CRAWL_METADATA {
        uuid id PK
        uuid document_id FK
        uuid url_id FK
        integer crawl_depth
        integer http_status
        timestamp crawled_at
        text user_agent
        jsonb headers
    }

    URLS {
        uuid id PK
        text url UK
        text domain
        text tld
        integer path_depth
        timestamp first_seen
        timestamp last_crawled
        integer crawl_count
        string status
    }

    CRAWL_QUEUE {
        uuid id PK
        uuid url_id FK
        float priority_score
        integer retry_count
        timestamp scheduled_for
        string status
        jsonb metadata
    }

    URL_CURIOSITY_SCORES {
        uuid id PK
        uuid url_id FK
        uuid model_id FK
        float curiosity_score
        float predicted_novelty
        timestamp scored_at
        jsonb features
    }

    NOVELTY_SCORES {
        uuid id PK
        uuid document_id FK
        uuid model_id FK
        float intrinsic_reward
        float novelty_score
        float final_score
        timestamp computed_at
        jsonb debug_info
    }

    TOPICS {
        uuid id PK
        string name
        text description
        jsonb keywords
        integer document_count
        timestamp created_at
        timestamp updated_at
    }

    TOPIC_EVOLUTION {
        uuid id PK
        uuid topic_id FK
        float divergence_score
        string divergence_type "JS or KL"
        timestamp measured_at
        jsonb distribution
    }

    CLUSTERS {
        uuid id PK
        string name
        integer cluster_number
        text centroid_description
        integer member_count
        timestamp created_at
    }

    CLUSTER_ASSIGNMENTS {
        uuid id PK
        uuid document_id FK
        uuid cluster_id FK
        float distance_to_centroid
        timestamp assigned_at
    }

    ICM_MODELS {
        uuid id PK
        string version UK
        string model_type "inverse|forward|ensemble"
        text model_path
        jsonb hyperparameters
        jsonb metrics
        timestamp trained_at
        boolean is_active
    }

    MODEL_CHECKPOINTS {
        uuid id PK
        uuid model_id FK
        integer epoch
        text checkpoint_path
        jsonb metrics
        timestamp created_at
    }
```

---

## 6. KUBERNETES DEPLOYMENT ARCHITECTURE

```mermaid
graph TB
    subgraph K8S["Kubernetes Cluster"]
        subgraph Namespace1["Namespace: fascinator-prod"]
            subgraph Ingress["Ingress Layer"]
                NGINX["NGINX Ingress<br/>Controller"]
                CERT["cert-manager<br/>TLS Certificates"]
            end

            subgraph API["API Tier"]
                APIPOD1["FastAPI Pod 1"]
                APIPOD2["FastAPI Pod 2"]
                APIPOD3["FastAPI Pod 3"]
                APISVC["API Service<br/>LoadBalancer"]

                APIPOD1 --- APISVC
                APIPOD2 --- APISVC
                APIPOD3 --- APISVC
            end

            subgraph Crawlers["Crawler Tier"]
                CPOD1["Crawler Pod 1<br/>Rust"]
                CPOD2["Crawler Pod 2<br/>Go"]
                CPOD3["Crawler Pod 3<br/>Rust"]
                CHPA["HPA: 3-10 replicas"]

                CPOD1 -.- CHPA
                CPOD2 -.- CHPA
                CPOD3 -.- CHPA
            end

            subgraph Processing["Processing Tier"]
                TPOD1["Tika Pod 1"]
                TPOD2["Tika Pod 2"]
                NPOD1["NLP Pod 1"]
                NPOD2["NLP Pod 2"]
                PHPA["HPA: 2-8 replicas"]

                TPOD1 -.- PHPA
                TPOD2 -.- PHPA
                NPOD1 -.- PHPA
                NPOD2 -.- PHPA
            end

            subgraph Intelligence["Intelligence Tier"]
                ICMPOD1["ICM Pod 1<br/>GPU"]
                ICMPOD2["ICM Pod 2<br/>GPU"]
                TOPICPOD["Topic Model Pod"]
                CLUSTERPOD["Cluster Pod"]

                GPU1["GPU Node Affinity"]
                ICMPOD1 -.- GPU1
                ICMPOD2 -.- GPU1
            end

            subgraph Orchestration["Orchestration Tier"]
                AIRFLOW["Airflow<br/>Scheduler"]
                AIRWEB["Airflow<br/>Webserver"]
                AIRWORK["Airflow<br/>Workers x3"]
            end

            subgraph Storage["Storage Tier"]
                PGPOD["PostgreSQL<br/>StatefulSet"]
                PGPVC["PVC: 1TB SSD"]

                REDISPOD1["Redis Master"]
                REDISPOD2["Redis Replica 1"]
                REDISPOD3["Redis Replica 2"]
                REDISSVC["Redis Service"]

                MQPOD1["RabbitMQ Node 1"]
                MQPOD2["RabbitMQ Node 2"]
                MQPOD3["RabbitMQ Node 3"]
                MQSVC["RabbitMQ Service"]

                PGPOD --- PGPVC
                REDISPOD1 --- REDISSVC
                REDISPOD2 --- REDISSVC
                REDISPOD3 --- REDISSVC
                MQPOD1 --- MQSVC
                MQPOD2 --- MQSVC
                MQPOD3 --- MQSVC
            end

            subgraph Monitoring["Monitoring Tier"]
                PROMPOD["Prometheus"]
                GRAFPOD["Grafana"]
                ELKPOD["ELK Stack"]
                ALERTPOD["Alertmanager"]
            end
        end

        subgraph PV["Persistent Volumes"]
            ZFS["ZFS Storage Pool<br/>2TB NVMe<br/>CSI Driver"]
            S3["S3/MinIO<br/>Object Storage<br/>Backup & Archive"]
        end
    end

    subgraph External["External Access"]
        USER["Users"]
        ADMIN["Administrators"]
    end

    %% External Access
    USER --> NGINX
    ADMIN --> NGINX
    NGINX --> APISVC
    NGINX --> AIRWEB
    NGINX --> GRAFPOD

    %% Internal Communication
    APISVC --> PGPOD
    APISVC --> REDISSVC

    CPOD1 --> MQSVC
    CPOD2 --> MQSVC
    CPOD3 --> MQSVC

    TPOD1 --> MQSVC
    TPOD2 --> MQSVC
    NPOD1 --> MQSVC
    NPOD2 --> MQSVC

    ICMPOD1 --> PGPOD
    ICMPOD2 --> PGPOD
    ICMPOD1 --> REDISSVC
    ICMPOD2 --> REDISSVC

    AIRFLOW --> MQSVC
    AIRWORK --> MQSVC

    %% Storage
    CPOD1 -.-> ZFS
    TPOD1 -.-> ZFS
    NPOD1 -.-> ZFS
    PGPOD -.-> ZFS

    ICMPOD1 -.-> S3
    PGPOD -.-> S3

    %% Monitoring
    PROMPOD --> APIPOD1
    PROMPOD --> CPOD1
    PROMPOD --> ICMPOD1
    PROMPOD --> PGPOD
    PROMPOD --> REDISPOD1
    PROMPOD --> MQPOD1

    GRAFPOD --> PROMPOD
    ALERTPOD --> PROMPOD

    classDef api fill:#4caf50,stroke:#2e7d32,color:#fff,stroke-width:2px
    classDef crawler fill:#2196f3,stroke:#1565c0,color:#fff,stroke-width:2px
    classDef processing fill:#ff9800,stroke:#e65100,color:#fff,stroke-width:2px
    classDef intelligence fill:#9c27b0,stroke:#6a1b9a,color:#fff,stroke-width:2px
    classDef storage fill:#f44336,stroke:#c62828,color:#fff,stroke-width:2px
    classDef monitoring fill:#009688,stroke:#004d40,color:#fff,stroke-width:2px

    class APIPOD1,APIPOD2,APIPOD3,APISVC api
    class CPOD1,CPOD2,CPOD3 crawler
    class TPOD1,TPOD2,NPOD1,NPOD2 processing
    class ICMPOD1,ICMPOD2,TOPICPOD,CLUSTERPOD intelligence
    class PGPOD,REDISPOD1,MQPOD1,ZFS,S3 storage
    class PROMPOD,GRAFPOD,ELKPOD,ALERTPOD monitoring
```

---

## 7. CI/CD PIPELINE

```mermaid
flowchart LR
    subgraph Developer["Developer Workflow"]
        CODE["Write Code"]
        COMMIT["Git Commit"]
        PUSH["Git Push"]
        PR["Create Pull Request"]

        CODE --> COMMIT --> PUSH --> PR
    end

    subgraph CI["Continuous Integration (GitHub Actions)"]
        TRIGGER["Trigger: PR/Push"]

        subgraph Tests["Test Stage"]
            LINT["Linting<br/>pylint, black, mypy"]
            UNIT["Unit Tests<br/>pytest"]
            SECURITY["Security Scan<br/>bandit, safety"]

            LINT -.-> UNIT -.-> SECURITY
        end

        subgraph Build["Build Stage"]
            DOCKER["Build Docker Images"]
            SCAN["Image Scanning<br/>Trivy"]
            PUSH_REG["Push to Registry"]

            DOCKER --> SCAN --> PUSH_REG
        end

        TRIGGER --> Tests
        Tests -->|Pass| Build
        Tests -->|Fail| NOTIFY1["Notify Failure"]
    end

    subgraph CD_Staging["CD: Staging Environment"]
        DEPLOY_STG["Deploy to Staging"]

        subgraph Integration["Integration Tests"]
            E2E["End-to-End Tests"]
            LOAD["Load Tests<br/>Locust"]
            PERF["Performance Tests"]

            E2E --> LOAD --> PERF
        end

        HEALTH_STG["Health Checks"]
        SMOKE["Smoke Tests"]

        DEPLOY_STG --> Integration
        Integration --> HEALTH_STG --> SMOKE
    end

    subgraph Approval["Manual Approval"]
        REVIEW["Human Review"]
        APPROVE["Approve Deploy"]

        REVIEW --> APPROVE
    end

    subgraph CD_Prod["CD: Production Environment"]
        BACKUP["Backup Current State"]

        subgraph Deploy["Deployment Strategy"]
            BLUE["Blue-Green Deploy"]
            CANARY["Canary Release<br/>5% → 50% → 100%"]

            BLUE -.Alternative.- CANARY
        end

        MIGRATE["Run Migrations"]
        HEALTH_PROD["Health Checks"]

        BACKUP --> Deploy
        Deploy --> MIGRATE --> HEALTH_PROD
    end

    subgraph Monitoring["Post-Deploy Monitoring"]
        METRICS["Monitor Metrics<br/>5 minutes"]
        ERROR["Error Rate Check"]
        LATENCY["Latency Check"]

        METRICS --> ERROR
        METRICS --> LATENCY
    end

    subgraph Rollback["Automatic Rollback"]
        DETECT["Detect Issues"]
        REVERT["Revert to Previous"]
        NOTIFY2["Alert Team"]

        DETECT --> REVERT --> NOTIFY2
    end

    %% Flow
    PR --> TRIGGER
    Build -->|Success| DEPLOY_STG
    SMOKE -->|Pass| REVIEW
    SMOKE -->|Fail| NOTIFY1
    APPROVE --> BACKUP
    HEALTH_PROD --> METRICS
    ERROR -->|High| DETECT
    LATENCY -->|High| DETECT
    ERROR -->|Normal| SUCCESS["✓ Deploy Complete"]
    LATENCY -->|Normal| SUCCESS

    style SUCCESS fill:#4caf50,stroke:#2e7d32,color:#fff,stroke-width:3px
    style NOTIFY1 fill:#f44336,stroke:#c62828,color:#fff,stroke-width:2px
    style NOTIFY2 fill:#f44336,stroke:#c62828,color:#fff,stroke-width:2px
    style APPROVE fill:#2196f3,stroke:#1565c0,color:#fff,stroke-width:2px
```

---

## 8. MONITORING & ALERTING ARCHITECTURE

```mermaid
graph TB
    subgraph Applications["Application Components"]
        API2["FastAPI"]
        CRAWL2["Crawlers"]
        PROCESS2["Processors"]
        ICM2["ICM Module"]
        PG2["PostgreSQL"]
        REDIS2["Redis"]
        MQ2["RabbitMQ"]
    end

    subgraph Metrics["Metrics Collection"]
        PROM_EXP1["Prometheus Exporter<br/>Python"]
        PROM_EXP2["PostgreSQL Exporter"]
        PROM_EXP3["Redis Exporter"]
        PROM_EXP4["RabbitMQ Exporter"]
        PROM_EXP5["Node Exporter"]
        PROM_EXP6["cAdvisor"]

        API2 --> PROM_EXP1
        CRAWL2 --> PROM_EXP1
        PROCESS2 --> PROM_EXP1
        ICM2 --> PROM_EXP1
        PG2 --> PROM_EXP2
        REDIS2 --> PROM_EXP3
        MQ2 --> PROM_EXP4
    end

    subgraph Logs["Log Collection"]
        FILEBEAT["Filebeat"]
        FLUENTD["Fluentd"]

        API2 -.-> FILEBEAT
        CRAWL2 -.-> FILEBEAT
        PROCESS2 -.-> FLUENTD
        ICM2 -.-> FLUENTD
    end

    subgraph Traces["Distributed Tracing"]
        OTEL["OpenTelemetry<br/>Collector"]

        API2 -.-> OTEL
        ICM2 -.-> OTEL
    end

    subgraph Storage_Monitoring["Monitoring Storage"]
        PROM2["Prometheus<br/>Time-Series DB"]
        ELASTIC["Elasticsearch<br/>Log Storage"]
        JAEGER["Jaeger<br/>Trace Storage"]

        PROM_EXP1 --> PROM2
        PROM_EXP2 --> PROM2
        PROM_EXP3 --> PROM2
        PROM_EXP4 --> PROM2
        PROM_EXP5 --> PROM2
        PROM_EXP6 --> PROM2

        FILEBEAT --> ELASTIC
        FLUENTD --> ELASTIC

        OTEL --> JAEGER
    end

    subgraph Visualization["Visualization Layer"]
        GRAF2["Grafana<br/>Dashboards"]
        KIBANA["Kibana<br/>Log Analytics"]
        JAEGER_UI["Jaeger UI<br/>Trace Viewer"]

        PROM2 --> GRAF2
        ELASTIC --> KIBANA
        JAEGER --> JAEGER_UI
    end

    subgraph Dashboards["Custom Dashboards"]
        DASH1["System Overview<br/>CPU, Memory, Disk"]
        DASH2["Crawl Performance<br/>Pages/sec, Success Rate"]
        DASH3["ICM Metrics<br/>Curiosity Distribution"]
        DASH4["Database Performance<br/>Query Latency, Connections"]
        DASH5["Queue Metrics<br/>Queue Depth, Consumer Lag"]

        GRAF2 --> DASH1
        GRAF2 --> DASH2
        GRAF2 --> DASH3
        GRAF2 --> DASH4
        GRAF2 --> DASH5
    end

    subgraph Alerting["Alerting System"]
        ALERT_MGR["Alertmanager"]

        subgraph Rules["Alert Rules"]
            RULE1["High Error Rate > 5%"]
            RULE2["Queue Backup > 10k"]
            RULE3["DB Connections > 90%"]
            RULE4["Disk Usage > 85%"]
            RULE5["Crawl Success < 70%"]
            RULE6["ICM Prediction Anomaly"]
        end

        PROM2 --> ALERT_MGR
        ALERT_MGR --> Rules
    end

    subgraph Notifications["Notification Channels"]
        SLACK["Slack"]
        EMAIL["Email"]
        PAGER["PagerDuty"]
        WEBHOOK["Webhook"]

        ALERT_MGR --> SLACK
        ALERT_MGR --> EMAIL
        ALERT_MGR --> PAGER
        ALERT_MGR --> WEBHOOK
    end

    subgraph OnCall["On-Call Response"]
        RUNBOOK["Runbook<br/>Automation"]
        INCIDENT["Incident<br/>Management"]

        PAGER --> RUNBOOK
        RUNBOOK --> INCIDENT
    end

    style PROM2 fill:#e6522c,stroke:#b71c1c,color:#fff,stroke-width:2px
    style GRAF2 fill:#f05a28,stroke:#d32f2f,color:#fff,stroke-width:2px
    style ALERT_MGR fill:#ffc107,stroke:#f57c00,color:#000,stroke-width:2px
    style PAGER fill:#d32f2f,stroke:#b71c1c,color:#fff,stroke-width:3px
```

---

## 9. CURIOSITY-DRIVEN EXPLORATION STATES

```mermaid
stateDiagram-v2
    [*] --> URLDiscovered

    URLDiscovered --> InQueue: Add to Frontier
    InQueue --> CuriosityScoring: Pop from Queue

    CuriosityScoring --> HighCuriosity: Score > Threshold
    CuriosityScoring --> LowCuriosity: Score ≤ Threshold

    LowCuriosity --> InQueue: Requeue with Lower Priority
    LowCuriosity --> Discarded: After N Low Scores

    HighCuriosity --> Crawling: Begin Crawl

    Crawling --> CrawlSuccess: HTTP 200
    Crawling --> CrawlFailure: HTTP Error/Timeout

    CrawlFailure --> RetryQueue: Retry Count < Max
    CrawlFailure --> Failed: Max Retries Exceeded
    RetryQueue --> Crawling: Exponential Backoff

    CrawlSuccess --> Processing: Extract Content

    Processing --> ContentAnalysis: Tika + NLP

    ContentAnalysis --> DocumentScoring: Generate Embeddings

    DocumentScoring --> HighNovelty: Novelty > Threshold
    DocumentScoring --> LowNovelty: Novelty ≤ Threshold

    HighNovelty --> Stored: Save to Hot Storage
    LowNovelty --> Evicted: Move to Cold Storage

    Stored --> URLExtraction: Extract Links

    URLExtraction --> URLDiscovered: New URLs Found
    URLExtraction --> ModelUpdate: No New URLs

    ModelUpdate --> [*]: Update ICM Weights

    Evicted --> Archived: ZFS Snapshot
    Archived --> [*]

    Failed --> [*]
    Discarded --> [*]

    note right of CuriosityScoring
        ICM Inverse + Forward Model
        Predict intrinsic reward
        Rank by exploration value
    end note

    note right of DocumentScoring
        ICM computes prediction error
        ||ŝ_t - s_t||² = novelty
        High error = high curiosity
    end note

    note right of HighNovelty
        Document teaches ICM
        about new patterns
        Updates model weights
    end note

    note right of LowNovelty
        Document too similar
        to existing corpus
        Low learning value
    end note
```

---

## 10. TOPIC MODEL EVOLUTION TRACKING

```mermaid
graph LR
    subgraph TimeT0["Time: T0 (Day 0)"]
        DOC_T0["Documents 1-1000"]
        TRAIN_T0["Train LDA Model"]
        TOPIC_T0["Topics: A, B, C"]
        DIST_T0["Topic Distributions<br/>P_T0(topics)"]

        DOC_T0 --> TRAIN_T0 --> TOPIC_T0 --> DIST_T0
    end

    subgraph TimeT1["Time: T1 (Day 7)"]
        DOC_T1["Documents 1001-2500"]
        RETRAIN_T1["Incremental Update"]
        TOPIC_T1["Topics: A', B', C', D"]
        DIST_T1["Topic Distributions<br/>P_T1(topics)"]

        DOC_T1 --> RETRAIN_T1 --> TOPIC_T1 --> DIST_T1
    end

    subgraph TimeT2["Time: T2 (Day 14)"]
        DOC_T2["Documents 2501-5000"]
        RETRAIN_T2["Incremental Update"]
        TOPIC_T2["Topics: A'', B'', C'', D', E"]
        DIST_T2["Topic Distributions<br/>P_T2(topics)"]

        DOC_T2 --> RETRAIN_T2 --> TOPIC_T2 --> DIST_T2
    end

    subgraph Divergence["Divergence Analysis"]
        JS_01["Jensen-Shannon<br/>Divergence<br/>D_JS(P_T0 || P_T1)"]
        KL_01["Kullback-Leibler<br/>Divergence<br/>D_KL(P_T0 || P_T1)"]

        JS_12["Jensen-Shannon<br/>Divergence<br/>D_JS(P_T1 || P_T2)"]
        KL_12["Kullback-Leibler<br/>Divergence<br/>D_KL(P_T1 || P_T2)"]

        DIST_T0 --> JS_01
        DIST_T1 --> JS_01
        DIST_T0 --> KL_01
        DIST_T1 --> KL_01

        DIST_T1 --> JS_12
        DIST_T2 --> JS_12
        DIST_T1 --> KL_12
        DIST_T2 --> KL_12
    end

    subgraph Actions["Adaptive Actions"]
        HIGH_DIV{High<br/>Divergence?}

        ACTION1["Increase Exploration<br/>Seek Novel Domains"]
        ACTION2["Retrain ICM<br/>Update Feature Space"]
        ACTION3["Adjust Eviction<br/>Threshold"]

        LOW_DIV["Low Divergence<br/>Corpus Saturation"]
        ACTION4["Expand Crawl Domains"]
        ACTION5["Adjust Curiosity Weights"]

        JS_12 --> HIGH_DIV
        HIGH_DIV -->|Yes| ACTION1
        HIGH_DIV -->|Yes| ACTION2
        HIGH_DIV -->|Yes| ACTION3

        HIGH_DIV -->|No| LOW_DIV
        LOW_DIV --> ACTION4
        LOW_DIV --> ACTION5
    end

    subgraph Feedback["Feedback Loop"]
        ACTION1 -.->|New Domains| DOC_T2
        ACTION2 -.->|Retrained Model| RETRAIN_T2
        ACTION4 -.->|Expanded Sources| DOC_T2
    end

    style JS_01 fill:#e1bee7,stroke:#8e24aa,stroke-width:2px
    style KL_01 fill:#e1bee7,stroke:#8e24aa,stroke-width:2px
    style JS_12 fill:#e1bee7,stroke:#8e24aa,stroke-width:2px
    style KL_12 fill:#e1bee7,stroke:#8e24aa,stroke-width:2px
    style ACTION1 fill:#81c784,stroke:#388e3c,stroke-width:2px
    style ACTION2 fill:#81c784,stroke:#388e3c,stroke-width:2px
    style ACTION3 fill:#81c784,stroke:#388e3c,stroke-width:2px
    style ACTION4 fill:#64b5f6,stroke:#1976d2,stroke-width:2px
    style ACTION5 fill:#64b5f6,stroke:#1976d2,stroke-width:2px
```

---

## 11. SECURITY ARCHITECTURE

```mermaid
graph TB
    subgraph Internet["Internet"]
        USER2["Users"]
        ATTACKER["Potential Attackers"]
    end

    subgraph EdgeSecurity["Edge Security"]
        WAF["Web Application Firewall<br/>ModSecurity"]
        DDOS["DDoS Protection<br/>Cloudflare/AWS Shield"]
        RATE["Rate Limiter"]

        USER2 --> DDOS --> WAF --> RATE
        ATTACKER -.X.-> DDOS
    end

    subgraph NetworkSecurity["Network Security"]
        LB["Load Balancer<br/>TLS Termination"]
        FW["Firewall Rules<br/>iptables/nftables"]
        VPN2["VPN Gateway<br/>Admin Access Only"]

        RATE --> LB
        LB --> FW
        VPN2 --> FW
    end

    subgraph ApplicationSecurity["Application Security"]
        AUTH["Authentication<br/>JWT Tokens"]
        AUTHZ["Authorization<br/>RBAC"]
        INPUT["Input Validation<br/>Pydantic"]
        CSRF2["CSRF Protection"]
        XSS2["XSS Prevention<br/>Content Security Policy"]

        FW --> AUTH
        AUTH --> AUTHZ
        AUTHZ --> INPUT
        INPUT --> CSRF2
        CSRF2 --> XSS2
    end

    subgraph DataSecurity["Data Security"]
        ENCRYPT_TRANSIT["Encryption in Transit<br/>TLS 1.3"]
        ENCRYPT_REST["Encryption at Rest<br/>LUKS/ZFS encryption"]
        SECRETS["Secrets Management<br/>HashiCorp Vault"]
        BACKUP_ENC["Encrypted Backups<br/>AES-256"]

        XSS2 --> ENCRYPT_TRANSIT
        ENCRYPT_TRANSIT --> ENCRYPT_REST
        SECRETS -.-> ENCRYPT_REST
        ENCRYPT_REST --> BACKUP_ENC
    end

    subgraph Compliance["Compliance & Privacy"]
        GDPR2["GDPR Compliance<br/>Data Retention"]
        AUDIT2["Audit Logging<br/>All Access Logged"]
        ANON["Data Anonymization<br/>PII Redaction"]
        ACCESS["Access Control<br/>Least Privilege"]

        BACKUP_ENC --> GDPR2
        GDPR2 --> AUDIT2
        AUDIT2 --> ANON
        ANON --> ACCESS
    end

    subgraph Monitoring_Security["Security Monitoring"]
        IDS["Intrusion Detection<br/>Suricata"]
        SIEM["SIEM<br/>Splunk/ELK"]
        VULN["Vulnerability Scanning<br/>Trivy/OWASP ZAP"]
        PENTEST["Penetration Testing<br/>Quarterly"]

        FW --> IDS
        IDS --> SIEM
        ApplicationSecurity --> VULN
        VULN --> PENTEST
    end

    subgraph IncidentResponse["Incident Response"]
        DETECT2["Detect<br/>Anomalies"]
        CONTAIN["Contain<br/>Isolate Systems"]
        ERADICATE["Eradicate<br/>Remove Threat"]
        RECOVER["Recover<br/>Restore Services"]
        LESSONS["Lessons Learned<br/>Post-Mortem"]

        SIEM --> DETECT2
        DETECT2 --> CONTAIN
        CONTAIN --> ERADICATE
        ERADICATE --> RECOVER
        RECOVER --> LESSONS
    end

    subgraph CrawlerEthics["Crawler Security & Ethics"]
        ROBOTS2["robots.txt<br/>Compliance"]
        USERAGENT["Proper User-Agent<br/>Identification"]
        RATELIMIT2["Domain Rate Limiting<br/>Respectful Crawling"]
        TOS["Terms of Service<br/>Compliance Check"]

        ROBOTS2 --> USERAGENT
        USERAGENT --> RATELIMIT2
        RATELIMIT2 --> TOS
    end

    style WAF fill:#f44336,stroke:#c62828,color:#fff,stroke-width:2px
    style DDOS fill:#f44336,stroke:#c62828,color:#fff,stroke-width:2px
    style AUTH fill:#4caf50,stroke:#2e7d32,color:#fff,stroke-width:2px
    style ENCRYPT_REST fill:#2196f3,stroke:#1565c0,color:#fff,stroke-width:2px
    style SECRETS fill:#9c27b0,stroke:#6a1b9a,color:#fff,stroke-width:2px
    style IDS fill:#ff9800,stroke:#e65100,color:#fff,stroke-width:2px
    style SIEM fill:#ff9800,stroke:#e65100,color:#fff,stroke-width:2px
```

---

## 12. SCALING STRATEGY

```mermaid
graph TB
    subgraph Metrics_Scaling["Scaling Metrics"]
        CPU_METRIC["CPU Usage > 70%"]
        MEM_METRIC["Memory > 80%"]
        QUEUE_METRIC["Queue Depth > 1000"]
        LATENCY_METRIC["API Latency > 500ms"]
        CRAWL_METRIC["Crawl Rate < Target"]
    end

    subgraph HPA_Controllers["Horizontal Pod Autoscalers"]
        HPA_API["API HPA<br/>Min: 3, Max: 10"]
        HPA_CRAWLER["Crawler HPA<br/>Min: 3, Max: 20"]
        HPA_PROCESS["Processor HPA<br/>Min: 2, Max: 15"]
        HPA_ICM["ICM HPA<br/>Min: 2, Max: 8"]

        CPU_METRIC --> HPA_API
        LATENCY_METRIC --> HPA_API

        QUEUE_METRIC --> HPA_CRAWLER
        CRAWL_METRIC --> HPA_CRAWLER

        QUEUE_METRIC --> HPA_PROCESS
        CPU_METRIC --> HPA_PROCESS

        MEM_METRIC --> HPA_ICM
        CPU_METRIC --> HPA_ICM
    end

    subgraph Actions_Scaling["Scaling Actions"]
        SCALE_UP["Scale Up<br/>Add Pods"]
        SCALE_DOWN["Scale Down<br/>Remove Pods"]

        HPA_API --> SCALE_UP
        HPA_CRAWLER --> SCALE_UP
        HPA_PROCESS --> SCALE_UP
        HPA_ICM --> SCALE_UP

        HPA_API -.->|Low Load| SCALE_DOWN
        HPA_CRAWLER -.->|Low Load| SCALE_DOWN
        HPA_PROCESS -.->|Low Load| SCALE_DOWN
    end

    subgraph Vertical["Vertical Scaling"]
        VPA["Vertical Pod Autoscaler"]
        RESOURCE_ADJUST["Adjust Pod<br/>CPU/Memory Requests"]

        MEM_METRIC --> VPA
        CPU_METRIC --> VPA
        VPA --> RESOURCE_ADJUST
    end

    subgraph Database_Scaling["Database Scaling"]
        PG_METRICS["DB Metrics<br/>Connection Pool Usage<br/>Query Latency"]

        READ_REPLICA["Add Read Replicas"]
        CONNECTION_POOL["Increase Connection Pool"]
        PARTITION["Table Partitioning"]
        CACHE_LAYER["Enhance Redis Cache"]

        PG_METRICS --> READ_REPLICA
        PG_METRICS --> CONNECTION_POOL
        PG_METRICS --> PARTITION
        PG_METRICS --> CACHE_LAYER
    end

    subgraph Storage_Scaling["Storage Scaling"]
        STORAGE_METRICS["Disk Usage > 75%"]

        ADD_DISK["Add Storage Volumes"]
        TIERING["Implement Tiering<br/>Hot → Warm → Cold"]
        EVICTION2["Aggressive Eviction"]
        COMPRESSION["Enable Compression"]

        STORAGE_METRICS --> ADD_DISK
        STORAGE_METRICS --> TIERING
        STORAGE_METRICS --> EVICTION2
        STORAGE_METRICS --> COMPRESSION
    end

    subgraph Cluster_Scaling["Cluster Scaling"]
        CLUSTER_METRICS["Node Resource<br/>Exhaustion"]

        ADD_NODES["Add K8s Nodes<br/>Cluster Autoscaler"]
        UPGRADE_NODES["Upgrade Node Types"]
        MULTI_REGION["Multi-Region<br/>Deployment"]

        CLUSTER_METRICS --> ADD_NODES
        CLUSTER_METRICS --> UPGRADE_NODES
        CLUSTER_METRICS -.->|Future| MULTI_REGION
    end

    subgraph Cost_Optimization["Cost Optimization"]
        SPOT["Use Spot Instances<br/>for Non-Critical"]
        SCHEDULE["Scheduled Scaling<br/>Down During Off-Peak"]
        RIGHT_SIZE["Right-Size Resources"]

        SCALE_DOWN --> SPOT
        SCALE_DOWN --> SCHEDULE
        VPA --> RIGHT_SIZE
    end

    style SCALE_UP fill:#4caf50,stroke:#2e7d32,color:#fff,stroke-width:3px
    style SCALE_DOWN fill:#ff9800,stroke:#e65100,color:#fff,stroke-width:2px
    style MULTI_REGION fill:#2196f3,stroke:#1565c0,color:#fff,stroke-width:2px
```

---

## Summary

These diagrams provide a comprehensive view of the proposed Fascinator system:

1. **High-Level Architecture**: Complete system overview with all layers
2. **Data Flow Pipeline**: Detailed step-by-step data processing
3. **ICM Workflow**: In-depth curiosity module operations
4. **Message Queue**: Task distribution architecture
5. **Database Schema**: Complete data model with relationships
6. **Kubernetes Deployment**: Production infrastructure
7. **CI/CD Pipeline**: Automated testing and deployment
8. **Monitoring**: Observability and alerting
9. **State Machine**: Curiosity-driven exploration states
10. **Topic Evolution**: Tracking content changes over time
11. **Security**: Comprehensive security layers
12. **Scaling**: Horizontal and vertical scaling strategies

Each diagram can be rendered using Mermaid-compatible tools (GitHub, GitLab, Mermaid Live Editor, etc.).
