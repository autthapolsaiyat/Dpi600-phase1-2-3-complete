#!/bin/bash
# ============================================================================
# DPI-600 Deployment Script
# Version: 2.0
# Date: December 2568
# ============================================================================

echo "🔬 DPI-600 Drug Profile Intelligence System"
echo "==========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check deployment target
if [ -z "$1" ]; then
    echo "Usage: ./deploy.sh [local|azure|nginx]"
    echo ""
    echo "Options:"
    echo "  local  - Start local development server"
    echo "  azure  - Deploy to Azure Static Web Apps"
    echo "  nginx  - Deploy to Nginx server"
    exit 1
fi

TARGET=$1

case $TARGET in
    "local")
        echo -e "${CYAN}🖥️  Starting local development server...${NC}"
        echo ""
        echo "Server will be available at: http://localhost:8000"
        echo "Press Ctrl+C to stop"
        echo ""
        python3 -m http.server 8000
        ;;
        
    "azure")
        echo -e "${CYAN}☁️  Deploying to Azure Static Web Apps...${NC}"
        echo ""
        
        # Check Azure CLI
        if ! command -v az &> /dev/null; then
            echo -e "${RED}❌ Azure CLI not installed${NC}"
            echo "Install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
            exit 1
        fi
        
        # Check login status
        az account show &> /dev/null
        if [ $? -ne 0 ]; then
            echo "Please login to Azure..."
            az login
        fi
        
        # Deploy
        echo "Deploying files..."
        
        # You need to configure these variables
        RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-dpi600-rg}"
        APP_NAME="${AZURE_APP_NAME:-dpi600-system}"
        LOCATION="${AZURE_LOCATION:-eastasia}"
        
        # Create resource group if not exists
        az group create --name $RESOURCE_GROUP --location $LOCATION 2>/dev/null
        
        # Deploy static web app
        az staticwebapp create \
            --name $APP_NAME \
            --resource-group $RESOURCE_GROUP \
            --source . \
            --location $LOCATION \
            --branch main \
            --app-location "/" \
            --output-location "/" \
            --login-with-github
        
        echo -e "${GREEN}✅ Deployment complete!${NC}"
        echo ""
        echo "Your app is available at:"
        az staticwebapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "defaultHostname" -o tsv
        ;;
        
    "nginx")
        echo -e "${CYAN}🌐 Deploying to Nginx server...${NC}"
        echo ""
        
        # Check sudo
        if [ "$EUID" -ne 0 ]; then
            echo "Please run with sudo: sudo ./deploy.sh nginx"
            exit 1
        fi
        
        # Set deploy path
        DEPLOY_PATH="/var/www/html/dpi600"
        
        # Create directory
        mkdir -p $DEPLOY_PATH
        
        # Copy files
        echo "Copying files to $DEPLOY_PATH..."
        cp -r ./* $DEPLOY_PATH/
        rm $DEPLOY_PATH/deploy.sh
        
        # Set permissions
        chown -R www-data:www-data $DEPLOY_PATH
        chmod -R 755 $DEPLOY_PATH
        
        # Restart Nginx
        systemctl restart nginx
        
        echo -e "${GREEN}✅ Deployment complete!${NC}"
        echo ""
        echo "Your app is available at: http://your-server/dpi600"
        ;;
        
    *)
        echo -e "${RED}❌ Unknown deployment target: $TARGET${NC}"
        echo "Use: local, azure, or nginx"
        exit 1
        ;;
esac

echo ""
echo "🔐 Demo Login: admin / admin123"
echo ""
