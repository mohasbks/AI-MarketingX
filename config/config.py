import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configurations
GOOGLE_ADS_CONFIG = {
    'client_id': os.getenv('GOOGLE_ADS_CLIENT_ID'),
    'client_secret': os.getenv('GOOGLE_ADS_CLIENT_SECRET'),
    'developer_token': os.getenv('GOOGLE_ADS_DEVELOPER_TOKEN'),
    'refresh_token': os.getenv('GOOGLE_ADS_REFRESH_TOKEN'),
    'login_customer_id': os.getenv('GOOGLE_ADS_LOGIN_CUSTOMER_ID')
}

FACEBOOK_ADS_CONFIG = {
    'app_id': os.getenv('FACEBOOK_APP_ID'),
    'app_secret': os.getenv('FACEBOOK_APP_SECRET'),
    'access_token': os.getenv('FACEBOOK_ACCESS_TOKEN')
}

# Database Configurations
MONGODB_CONFIG = {
    'connection_string': os.getenv('MONGODB_CONNECTION_STRING'),
    'database_name': os.getenv('MONGODB_DATABASE_NAME')
}

# Model Configurations
MODEL_CONFIG = {
    'performance_predictor': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'validation_split': 0.2
    },
    'audience_analyzer': {
        'embedding_dim': 64,
        'num_heads': 4,
        'dropout_rate': 0.2
    }
}

# Feature Engineering Configurations
FEATURE_CONFIG = {
    'numerical_features': [
        'daily_spend',
        'impressions',
        'clicks',
        'conversions',
        'ctr',
        'cpc'
    ],
    'categorical_features': [
        'campaign_type',
        'platform',
        'targeting_type',
        'device_type'
    ]
}

# API Endpoints Configuration
API_CONFIG = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', 8000)),
    'debug': os.getenv('DEBUG', 'False').lower() == 'true'
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'app.log',
            'mode': 'a'
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
