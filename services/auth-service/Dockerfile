FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy TypeScript config and source
COPY tsconfig.json ./
COPY src ./src

# Install TypeScript and build dependencies
RUN npm install -g typescript && \
    npm install --only=dev && \
    npm run build && \
    npm prune --production

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S platform3 -u 1001

# Change ownership of the app directory
RUN chown -R platform3:nodejs /app
USER platform3

# Expose port
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3001/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1) })"

# Start the application
CMD ["npm", "start"]
