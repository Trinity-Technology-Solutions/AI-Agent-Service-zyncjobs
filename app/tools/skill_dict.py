SKILL_DICT = {
    "languages": {
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "C", "Go", "Golang",
        "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "Perl", "R", "MATLAB",
        "Dart", "Flutter", "Shell", "Bash", "SQL", "PL/SQL", "T-SQL", "HTML", "CSS",
        "Sass", "Less", "COBOL", "Fortran", "Lua", "Haskell", "Clojure", "Elixir",
        "Objective-C", "Assembly", "Groovy", "Solidity", "VBA", "PowerShell",
        "Delphi", "Pascal", "Julia", "Verilog", "VHDL", "ABAP",
    },
    "frameworks": {
        "React", "Angular", "Vue", "Next.js", "Nuxt", "Svelte", "Express", "Django",
        "Flask", "FastAPI", "Spring", "Spring Boot", "Laravel", "Ruby on Rails",
        "ASP.NET", ".NET Core", "Node.js", "Deno", "Bootstrap", "Tailwind CSS",
        "jQuery", "Redux", "GraphQL", "Apollo", "Prisma", "TypeORM", "Sequelize",
        "Mongoose", "Hibernate", "MyBatis", "PyTorch", "TensorFlow", "Keras",
        "Scikit-learn", "Pandas", "NumPy", "OpenCV", "NLTK", "spaCy", "Hugging Face",
        "LangChain", "LlamaIndex", "FastAPI", "Flask", "Django REST", "ASP.NET MVC",
        "Vite", "Webpack", "Gulp", "Grunt", "Electron", "React Native", "Ionic",
        "Xamarin", "Unity", "Unreal Engine", "OpenGL", "WebGL", "D3.js",
        "Chart.js", "Three.js", "Mocha", "Jest", "Cypress", "Playwright",
        "Puppeteer", "Selenium", "JUnit", "pytest", "Maven", "Gradle",
    },
    "databases": {
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle", "SQL Server", "SQLite",
        "MariaDB", "Cassandra", "DynamoDB", "Firebase", "Firestore", "CouchDB",
        "Elasticsearch", "Neo4j", "InfluxDB", "TimescaleDB", "ClickHouse",
        "Snowflake", "BigQuery", "Redshift", "CockroachDB", "Supabase",
        "PlanetScale", "MSSQL", "Db2", "HBase", "Memcached", "Aurora",
    },
    "cloud": {
        "AWS", "Amazon Web Services", "Azure", "Microsoft Azure", "GCP",
        "Google Cloud Platform", "Google Cloud", "Heroku", "DigitalOcean",
        "Linode", "Vercel", "Netlify", "Cloudflare", "Terraform", "Pulumi",
        "Ansible", "Chef", "Puppet", "Docker", "Kubernetes", "K8s", "ECS",
        "EKS", "Fargate", "Lambda", "Cloud Functions", "S3", "EC2", "RDS",
        "CloudFormation", "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI",
        "CircleCI", "Travis CI", "ArgoCD", "Helm", "Istio", "Envoy",
    },
    "tools": {
        "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence", "Slack",
        "Trello", "Asana", "Notion", "Figma", "Sketch", "Adobe XD", "Photoshop",
        "Illustrator", "Canva", "VS Code", "IntelliJ", "Eclipse", "PyCharm",
        "WebStorm", "Vim", "Emacs", "Sublime Text", "Postman", "Insomnia",
        "Swagger", "OpenAPI", "Kafka", "RabbitMQ", "Nginx", "Apache", "Tomcat",
        "Linux", "Ubuntu", "CentOS", "Windows Server", "macOS", "Bash", "Zsh",
        "Wireshark", "Fiddler", "Charles", "New Relic", "Datadog", "Grafana",
        "Prometheus", "Sentry", "ELK Stack", "Logstash", "Kibana", "Splunk",
        "Tableau", "Power BI", "Excel", "Jupyter", "Colab", "Agile", "Scrum",
        "Kanban", "SAFe", "Waterfall", "REST", "RESTful", "SOAP", "gRPC",
        "WebSocket", "OAuth", "JWT", "SAML", "LDAP", "SSH", "SSL", "TLS",
        "YAML", "JSON", "XML", "TOML", "Markdown", "LaTeX", "UML",
        "Stripe", "PayPal", "Twilio", "SendGrid", "AWS SES", "SQS", "SNS",
        "Microservices", "Serverless", "Monolith", "Event-Driven", "CQRS",
        "DDD", "TDD", "BDD", "SOLID", "Design Patterns", "Clean Architecture",
        "Hexagonal Architecture", "Onion Architecture", "MVC", "MVP", "MVVM",
        "NPM", "Yarn", "pnpm", "Pip", "Conda", "Docker Compose", "Vagrant",
        "Webpack", "Parcel", "Babel", "ESLint", "Prettier", "Husky",
        "Nginx", "Apache", "HAProxy", "Traefik", "Caddy",
    },
}


ALL_SKILLS: set[str] = set()
for category in SKILL_DICT.values():
    ALL_SKILLS.update(category)

SKILL_LOWERCASE: dict[str, str] = {s.lower(): s for s in ALL_SKILLS}


def filter_skills(raw_skills: list[str]) -> list[str]:
    matched: set[str] = set()
    for raw in raw_skills:
        raw_lower = raw.strip().lower()
        if raw_lower in SKILL_LOWERCASE:
            matched.add(SKILL_LOWERCASE[raw_lower])
            continue
        for skill_lower, skill_original in SKILL_LOWERCASE.items():
            if len(skill_lower) > 3 and (skill_lower in raw_lower or raw_lower in skill_lower):
                if len(raw_lower) <= len(skill_lower) + 5:
                    matched.add(skill_original)
                    break
    return sorted(matched, key=lambda s: s.lower())


def categorize_skills(skills: list[str]) -> dict:
    result = {
        "programming": [],
        "frameworks": [],
        "databases": [],
        "cloud": [],
        "tools": [],
    }
    category_map: dict[str, list[str]] = {}
    for cat_name, cat_skills in SKILL_DICT.items():
        for s in cat_skills:
            category_map[s.lower()] = cat_name
    for skill in skills:
        key = skill.lower()
        if key in category_map:
            cat = category_map[key]
            if cat == "languages":
                result["programming"].append(skill)
            else:
                result[cat].append(skill)
        else:
            result["tools"].append(skill)
    return result
