# Docker Complete Guide

## Docker Fundamentals

### What is Docker?
Docker is a containerization platform that packages applications with their dependencies into lightweight, portable containers.

**Key Concepts:**
- **Image**: Read-only template with instructions for creating containers
- **Container**: Running instance of an image
- **Dockerfile**: Text file with instructions to build an image
- **Registry**: Storage for Docker images (Docker Hub, ECR, etc.)

### Basic Commands
```bash
# Version and system info
docker --version
docker info
docker system df              # Show disk usage

# Image operations
docker images                 # List local images
docker pull nginx:latest      # Download image from registry
docker rmi image_name         # Remove image
docker image prune           # Remove unused images

# Container operations
docker ps                    # List running containers
docker ps -a                 # List all containers
docker run nginx            # Create and start container
docker stop container_id     # Stop container
docker rm container_id       # Remove container
docker exec -it container_id bash  # Enter container shell
```

## Dockerfile

### Basic Dockerfile Structure
```dockerfile
# Use official base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application code
COPY . .

# Expose port
EXPOSE 3000

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001
USER nextjs

# Set environment
ENV NODE_ENV=production

# Define startup command
CMD ["npm", "start"]
```

### Multi-stage Build
```dockerfile
# Build stage
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage
FROM node:18-alpine AS production

WORKDIR /app

# Copy only production dependencies
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy built application from builder stage
COPY --from=builder /app/dist ./dist

# Create user and set permissions
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001
USER nextjs

EXPOSE 3000

CMD ["node", "dist/index.js"]
```

### Python Dockerfile Example
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Docker Compose

### Basic docker-compose.yml
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgres://user:password@db:5432/myapp
    depends_on:
      - db
      - redis
    volumes:
      - .:/app
      - /app/node_modules
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    name: myapp-network
```

### Advanced Compose Features
```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
      args:
        NODE_ENV: production
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - JWT_SECRET=${JWT_SECRET}
    env_file:
      - .env.production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    networks:
      - backend

  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    command: ["python", "worker.py"]
    depends_on:
      db:
        condition: service_healthy
    scale: 3
    networks:
      - backend

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backup:/backup
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend

networks:
  backend:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/data/postgres
```

## Docker Commands Deep Dive

### Container Management
```bash
# Run containers with various options
docker run -d -p 8080:80 --name webserver nginx
docker run -it --rm ubuntu:20.04 bash          # Interactive, auto-remove
docker run -d --restart unless-stopped redis   # Auto-restart policy
docker run -e NODE_ENV=production myapp        # Environment variable
docker run -v $(pwd):/app -w /app node:18 npm install  # Mount volume, set workdir

# Container lifecycle
docker start container_name                     # Start stopped container
docker stop container_name                      # Gracefully stop
docker kill container_name                      # Force stop
docker restart container_name                   # Restart container
docker pause container_name                     # Pause processes
docker unpause container_name                   # Resume processes

# Container inspection
docker logs container_name                      # View logs
docker logs -f --tail 100 container_name      # Follow logs, last 100 lines
docker inspect container_name                   # Detailed info
docker stats                                    # Real-time resource usage
docker top container_name                      # Running processes
```

### Image Management
```bash
# Build images
docker build -t myapp:latest .                 # Build from Dockerfile
docker build -t myapp:v1.2 -f Dockerfile.prod .  # Custom Dockerfile
docker build --no-cache -t myapp .            # Build without cache
docker build --target production -t myapp:prod .  # Multi-stage build target

# Tag and push images
docker tag myapp:latest registry.example.com/myapp:latest
docker push registry.example.com/myapp:latest

# Save and load images
docker save -o myapp.tar myapp:latest         # Export image to tar
docker load -i myapp.tar                      # Import image from tar
docker export container_name > container.tar   # Export container filesystem
```

### Volume and Network Management
```bash
# Volume operations
docker volume create myvolume                  # Create named volume
docker volume ls                              # List volumes
docker volume inspect myvolume               # Volume details
docker volume rm myvolume                    # Remove volume
docker volume prune                          # Remove unused volumes

# Network operations
docker network create mynetwork              # Create network
docker network ls                           # List networks
docker network inspect bridge               # Network details
docker network connect mynetwork container  # Connect container to network
docker network disconnect mynetwork container  # Disconnect from network
```

## Security Best Practices

### Dockerfile Security
```dockerfile
# Use specific tags, not latest
FROM node:18.17.1-alpine

# Update package manager and remove cache
RUN apk update && apk upgrade && apk add --no-cache \
    curl \
    && rm -rf /var/cache/apk/*

# Create non-root user early
RUN addgroup -g 1001 -S nodejs \
    && adduser -S nextjs -u 1001 -G nodejs

# Set working directory
WORKDIR /app

# Copy package files first (for better caching)
COPY --chown=nextjs:nodejs package*.json ./

# Install dependencies as root if needed, then switch user
RUN npm ci --only=production

# Copy application files with correct ownership
COPY --chown=nextjs:nodejs . .

# Switch to non-root user
USER nextjs

# Don't run as PID 1 in containers
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "server.js"]

# Use HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1
```

### Security Scanning
```bash
# Scan images for vulnerabilities
docker scout quickview                        # Quick security overview
docker scout cves image_name                 # List CVEs
docker scout recommendations image_name      # Get security recommendations

# Using Trivy (external tool)
trivy image nginx:latest                     # Scan for vulnerabilities
trivy fs .                                   # Scan local filesystem
```

### Runtime Security
```bash
# Run with security options
docker run --security-opt=no-new-privileges myapp
docker run --read-only --tmpfs /tmp myapp   # Read-only filesystem
docker run --cap-drop=ALL --cap-add=CHOWN myapp  # Minimal capabilities
docker run --user 1001:1001 myapp          # Explicit user ID

# Resource limits
docker run --memory=512m --cpus="1.5" myapp
docker run --ulimit nofile=1024:1024 myapp
```

## Docker Compose Commands

### Service Management
```bash
# Start services
docker-compose up                           # Start in foreground
docker-compose up -d                        # Start in background
docker-compose up --build                   # Rebuild images before starting
docker-compose up service_name             # Start specific service

# Stop and clean up
docker-compose down                         # Stop and remove containers
docker-compose down -v                      # Also remove volumes
docker-compose down --rmi all               # Also remove images
docker-compose stop                         # Stop without removing

# Service operations
docker-compose restart service_name         # Restart service
docker-compose pause service_name          # Pause service
docker-compose unpause service_name        # Unpause service
docker-compose kill service_name           # Force stop service

# Scaling
docker-compose up --scale web=3 --scale worker=2
```

### Debugging and Logs
```bash
# View logs
docker-compose logs                         # All services
docker-compose logs web                     # Specific service
docker-compose logs -f --tail=100 web      # Follow logs

# Execute commands
docker-compose exec web bash               # Shell into running service
docker-compose run --rm web npm test       # Run one-off command

# Service status
docker-compose ps                           # List services
docker-compose top                          # Show running processes
```

## Production Deployment

### Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml myapp

# Swarm management
docker service ls                           # List services
docker service logs myapp_web             # Service logs
docker service scale myapp_web=5          # Scale service
docker service update --image myapp:v2 myapp_web  # Update service

# Node management
docker node ls                             # List swarm nodes
docker node update --availability drain node1  # Drain node for maintenance
```

### Docker Compose Production
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  web:
    image: registry.example.com/myapp:${VERSION}
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
    environment:
      - NODE_ENV=production
    env_file:
      - .env.production
    networks:
      - webnet
    secrets:
      - jwt_secret
      - db_password

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ssl_certs:/etc/nginx/ssl:ro
    deploy:
      placement:
        constraints: [node.role == manager]
    networks:
      - webnet

networks:
  webnet:
    external: true

secrets:
  jwt_secret:
    external: true
  db_password:
    external: true

volumes:
  ssl_certs:
    external: true
```

## Troubleshooting

### Common Issues and Solutions
```bash
# Container won't start
docker logs container_name                  # Check logs for errors
docker inspect container_name              # Check configuration
docker events                             # Monitor Docker events

# Permission issues
docker exec -it container_name ls -la     # Check file permissions
# Fix: Update Dockerfile USER directive or file ownership

# Port conflicts
docker ps                                  # Check port usage
netstat -tulpn | grep :8080               # Check system port usage
# Fix: Use different port mapping

# Out of disk space
docker system df                          # Check disk usage
docker system prune                       # Clean up unused resources
docker image prune                        # Remove unused images
docker container prune                    # Remove stopped containers

# Network connectivity issues
docker network ls                         # List networks
docker exec -it container_name ping other_container
docker exec -it container_name nslookup db  # Check DNS resolution
```

### Performance Optimization
```dockerfile
# Multi-stage build for smaller images
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 3000
CMD ["node", "dist/index.js"]

# .dockerignore to reduce build context
# .dockerignore
node_modules
npm-debug.log
.git
.gitignore
README.md
.env
.nyc_output
coverage
.nyc_output
.coverage
.coverage/
```

### Monitoring and Logging
```bash
# Resource monitoring
docker stats                              # Real-time resource usage
docker system events                      # System events stream

# Log management
docker logs --since 2023-01-01T00:00:00 container_name
docker logs --until 2023-12-31T23:59:59 container_name

# Container inspection
docker diff container_name                # Changes to filesystem
docker port container_name               # Port mappings
docker exec container_name env           # Environment variables
```