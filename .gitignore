import os
import logging
import sqlite3
import json
import requests
import feedparser
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import google.generativeai as genai
from cryptography.fernet import Fernet
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# API Keys and URLs
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
NOWPAYMENTS_API_KEY = os.getenv('NOWPAYMENTS_API_KEY')

# API URLs
COINPAPRIKA_API_URL = 'https://api.coinpaprika.com/v1/tickers?quotes=USD'
BINANCE_API_URL = 'https://api.binance.com/api/v3/klines'
CRYPTOPANIC_API_URL = 'https://cryptopanic.com/api/v1/posts/'
COINTELEGRAPH_RSS_URL = 'https://cointelegraph.com/rss'
COINMARKETCAL_API_URL = 'https://api.coinmarketcal.com/'
NOWPAYMENTS_API_URL = 'https://api.nowpayments.io/v1/invoice'

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Cache for Gemini responses
gemini_cache = {}

# Database setup
class Database:
    def __init__(self):
        self.conn = sqlite3.connect('crypto_bot.db')
        self.cursor = self.conn.cursor()
        self.setup_tables()
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    def setup_tables(self):
        # Users table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                language TEXT DEFAULT 'en',
                is_premium BOOLEAN DEFAULT 0,
                premium_expiry DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                daily_news_count INTEGER DEFAULT 0,
                daily_analysis_count INTEGER DEFAULT 0,
                daily_alert_count INTEGER DEFAULT 0,
                settings TEXT
            )
        ''')

        # Price alerts table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                coin TEXT,
                target_price REAL,
                is_triggered BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # Portfolios table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                portfolio_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # Payments table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS payments (
                payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                order_id TEXT UNIQUE,
                amount REAL,
                currency TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # Market sentiment table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_sentiment (
                sentiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT,
                sentiment_score REAL,
                analysis TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.conn.commit()

    def add_user(self, user_id: int, username: str, language: str = 'en'):
        try:
            self.cursor.execute('''
                INSERT OR IGNORE INTO users (user_id, username, language)
                VALUES (?, ?, ?)
            ''', (user_id, username, language))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error adding user: {e}")
            return False

    def get_user(self, user_id: int):
        self.cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        return self.cursor.fetchone()

    def update_user_language(self, user_id: int, language: str):
        self.cursor.execute('''
            UPDATE users SET language = ? WHERE user_id = ?
        ''', (language, user_id))
        self.conn.commit()

    def update_user_premium(self, user_id: int, is_premium: bool, expiry_date: datetime):
        self.cursor.execute('''
            UPDATE users 
            SET is_premium = ?, premium_expiry = ?
            WHERE user_id = ?
        ''', (is_premium, expiry_date, user_id))
        self.conn.commit()

    def add_price_alert(self, user_id: int, coin: str, target_price: float):
        self.cursor.execute('''
            INSERT INTO price_alerts (user_id, coin, target_price)
            VALUES (?, ?, ?)
        ''', (user_id, coin, target_price))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_user_alerts(self, user_id: int):
        self.cursor.execute('''
            SELECT * FROM price_alerts 
            WHERE user_id = ? AND is_triggered = 0
        ''', (user_id,))
        return self.cursor.fetchall()

    def update_alert_status(self, alert_id: int, is_triggered: bool):
        self.cursor.execute('''
            UPDATE price_alerts 
            SET is_triggered = ? 
            WHERE alert_id = ?
        ''', (is_triggered, alert_id))
        self.conn.commit()

    def save_portfolio(self, user_id: int, portfolio_data: dict):
        encrypted_data = self.cipher_suite.encrypt(json.dumps(portfolio_data).encode())
        self.cursor.execute('''
            INSERT OR REPLACE INTO portfolios (user_id, portfolio_data)
            VALUES (?, ?)
        ''', (user_id, encrypted_data))
        self.conn.commit()

    def get_portfolio(self, user_id: int):
        self.cursor.execute('SELECT portfolio_data FROM portfolios WHERE user_id = ?', (user_id,))
        result = self.cursor.fetchone()
        if result:
            decrypted_data = self.cipher_suite.decrypt(result[0])
            return json.loads(decrypted_data.decode())
        return None

    def add_payment(self, user_id: int, order_id: str, amount: float, currency: str, status: str):
        self.cursor.execute('''
            INSERT INTO payments (user_id, order_id, amount, currency, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, order_id, amount, currency, status))
        self.conn.commit()

    def get_payment(self, order_id: str):
        self.cursor.execute('SELECT * FROM payments WHERE order_id = ?', (order_id,))
        return self.cursor.fetchone()

    def update_daily_counts(self, user_id: int, count_type: str):
        column = f"daily_{count_type}_count"
        self.cursor.execute(f'''
            UPDATE users 
            SET {column} = {column} + 1 
            WHERE user_id = ?
        ''', (user_id,))
        self.conn.commit()

    def reset_daily_counts(self):
        self.cursor.execute('''
            UPDATE users 
            SET daily_news_count = 0,
                daily_analysis_count = 0,
                daily_alert_count = 0
        ''')
        self.conn.commit()

    def update_last_active(self, user_id: int):
        self.cursor.execute('''
            UPDATE users 
            SET last_active = CURRENT_TIMESTAMP 
            WHERE user_id = ?
        ''', (user_id,))
        self.conn.commit()

    def save_market_sentiment(self, coin: str, sentiment_score: float, analysis: str):
        self.cursor.execute('''
            INSERT INTO market_sentiment (coin, sentiment_score, analysis)
            VALUES (?, ?, ?)
        ''', (coin, sentiment_score, analysis))
        self.conn.commit()

    def get_latest_sentiment(self, coin: str):
        self.cursor.execute('''
            SELECT * FROM market_sentiment 
            WHERE coin = ? 
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (coin,))
        return self.cursor.fetchone()

    def close(self):
        self.conn.close()

# Initialize database
db = Database()

# Utility functions
def get_crypto_prices():
    """Get top 15 cryptocurrencies prices from Coinpaprika"""
    try:
        response = requests.get(COINPAPRIKA_API_URL)
        data = response.json()
        return sorted(data, key=lambda x: x['market_cap_usd'], reverse=True)[:15]
    except Exception as e:
        print(f"Error fetching crypto prices: {e}")
        return None

def get_binance_chart(symbol: str, interval: str, limit: int = 100):
    """Get candlestick data from Binance"""
    try:
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        response = requests.get(BINANCE_API_URL, params=params)
        data = response.json()
        return data
    except Exception as e:
        print(f"Error fetching Binance chart: {e}")
        return None

def create_candlestick_chart(data, symbol: str, interval: str):
    """Create candlestick chart using matplotlib"""
    try:
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['close'], label='Close Price')
        plt.title(f'{symbol} Price Chart ({interval})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        
        plt.xticks(rotation=45)
        
        chart_path = f'chart_{symbol}_{interval}.png'
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        
        return chart_path
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

def get_crypto_news():
    """Get crypto news from CryptoPanic and Cointelegraph"""
    news = []
    
    try:
        response = requests.get(CRYPTOPANIC_API_URL)
        data = response.json()
        for article in data.get('results', [])[:5]:
            news.append({
                'title': article['title'],
                'summary': article.get('summary', ''),
                'url': article['url'],
                'image': article.get('image', ''),
                'source': 'CryptoPanic'
            })
    except Exception as e:
        print(f"Error fetching CryptoPanic news: {e}")
    
    try:
        feed = feedparser.parse(COINTELEGRAPH_RSS_URL)
        for entry in feed.entries[:5]:
            image = ''
            if 'media_content' in entry:
                image = entry.media_content[0]['url']
            elif 'enclosures' in entry:
                image = entry.enclosures[0]['url']
            
            news.append({
                'title': entry.title,
                'summary': entry.get('summary', ''),
                'url': entry.link,
                'image': image,
                'source': 'Cointelegraph'
            })
    except Exception as e:
        print(f"Error fetching Cointelegraph news: {e}")
    
    return news

def get_crypto_events():
    """Get upcoming crypto events from CoinMarketCal"""
    try:
        response = requests.get(COINMARKETCAL_API_URL)
        data = response.json()
        return data.get('events', [])
    except Exception as e:
        print(f"Error fetching crypto events: {e}")
        return None

def analyze_with_gemini(prompt: str, language: str = 'en'):
    """Analyze text using Gemini API with caching"""
    cache_key = f"{prompt}_{language}"
    
    if cache_key in gemini_cache:
        cache_time, response = gemini_cache[cache_key]
        if datetime.now() - cache_time < timedelta(seconds=600):  # 10 minutes cache
            return response
    
    try:
        full_prompt = f"Please provide the response in {language}:\n{prompt}"
        response = model.generate_content(full_prompt)
        
        gemini_cache[cache_key] = (datetime.now(), response.text)
        
        return response.text
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return None

def format_price_change(price_change: float):
    """Format price change with emoji"""
    if price_change > 0:
        return f"📈 +{price_change:.2f}%"
    elif price_change < 0:
        return f"📉 {price_change:.2f}%"
    else:
        return "➡️ 0.00%"

def format_large_number(number: float):
    """Format large numbers with K, M, B suffixes"""
    if number >= 1e9:
        return f"${number/1e9:.2f}B"
    elif number >= 1e6:
        return f"${number/1e6:.2f}M"
    elif number >= 1e3:
        return f"${number/1e3:.2f}K"
    else:
        return f"${number:.2f}"

def create_inline_keyboard(options: list, callback_prefix: str):
    """Create inline keyboard for Telegram"""
    keyboard = []
    for i in range(0, len(options), 2):
        row = []
        row.append({
            'text': options[i],
            'callback_data': f"{callback_prefix}_{i}"
        })
        if i + 1 < len(options):
            row.append({
                'text': options[i + 1],
                'callback_data': f"{callback_prefix}_{i+1}"
            })
        keyboard.append(row)
    return keyboard

# Bot command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command"""
    user = update.effective_user
    db.add_user(user.id, user.username)
    
    keyboard = create_inline_keyboard(
        ['فارسی', 'English', 'العربية', 'Türkçe', 'Español'],
        'lang'
    )
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Welcome! Please select your language:\n"
        "خوش آمدید! لطفاً زبان خود را انتخاب کنید:\n"
        "مرحباً! الرجاء اختيار لغتك:\n"
        "Hoş geldiniz! Lütfen dilinizi seçin:\n"
        "¡Bienvenido! Por favor, seleccione su idioma:",
        reply_markup=reply_markup
    )

async def language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle language selection callback"""
    query = update.callback_query
    await query.answer()
    
    lang_index = int(query.data.split('_')[1])
    selected_lang = ['fa', 'en', 'ar', 'tr', 'es'][lang_index]
    
    db.update_user_language(query.from_user.id, selected_lang)
    
    # Main menu buttons
    keyboard = [
        [InlineKeyboardButton("📰 اخبار", callback_data='news'),
         InlineKeyboardButton("💰 قیمت‌ها", callback_data='prices')],
        [InlineKeyboardButton("📊 نمودارها", callback_data='charts'),
         InlineKeyboardButton("🎯 هشدار قیمت", callback_data='alerts')],
        [InlineKeyboardButton("🤖 تحلیل بازار", callback_data='analysis'),
         InlineKeyboardButton("📈 تحلیل تکنیکال", callback_data='technical')],
        [InlineKeyboardButton("⚖️ مقایسه ارزها", callback_data='compare'),
         InlineKeyboardButton("🔔 هشدار اخبار", callback_data='newsalert')],
        [InlineKeyboardButton("💼 پورتفولیو", callback_data='portfolio'),
         InlineKeyboardButton("📊 سود و زیان", callback_data='pnl')],
        [InlineKeyboardButton("📚 آموزش", callback_data='learn'),
         InlineKeyboardButton("🔍 بررسی کلاهبرداری", callback_data='scam')],
        [InlineKeyboardButton("💎 پرمیوم", callback_data='premium')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "منوی اصلی / Main Menu / القائمة الرئيسية / Ana Menü / Menú Principal",
        reply_markup=reply_markup
    )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all callback queries"""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'news':
        await handle_news(update, context)
    elif query.data == 'prices':
        await handle_prices(update, context)
    elif query.data == 'charts':
        await query.edit_message_text(
            "برای مشاهده نمودار، از دستور زیر استفاده کنید:\n"
            "Use the following command to view charts:\n"
            "/chart <symbol> <timeframe>\n"
            "Example: /chart BTCUSDT 1h"
        )
    elif query.data == 'alerts':
        await query.edit_message_text(
            "برای تنظیم هشدار قیمت، از دستور زیر استفاده کنید:\n"
            "Use the following command to set price alerts:\n"
            "/alert <symbol> <price>\n"
            "Example: /alert BTC 70000"
        )
    elif query.data == 'analysis':
        await query.edit_message_text(
            "برای تحلیل بازار، از دستور زیر استفاده کنید:\n"
            "Use the following command for market analysis:\n"
            "/sentiment <symbol>\n"
            "Example: /sentiment BTC"
        )
    elif query.data == 'technical':
        await query.edit_message_text(
            "برای تحلیل تکنیکال، از دستور زیر استفاده کنید:\n"
            "Use the following command for technical analysis:\n"
            "/technical <symbol> <timeframe>\n"
            "Example: /technical BTC 1d"
        )
    elif query.data == 'compare':
        await query.edit_message_text(
            "برای مقایسه ارزها، از دستور زیر استفاده کنید:\n"
            "Use the following command to compare coins:\n"
            "/compare <coin1> <coin2>\n"
            "Example: /compare BTC ETH"
        )
    elif query.data == 'newsalert':
        await query.edit_message_text(
            "برای دریافت هشدار اخبار، از دستور زیر استفاده کنید:\n"
            "Use the following command for news alerts:\n"
            "/newsalert <keyword>\n"
            "Example: /newsalert Bitcoin"
        )
    elif query.data == 'portfolio':
        await query.edit_message_text(
            "برای مدیریت پورتفولیو، از دستور زیر استفاده کنید:\n"
            "Use the following command to manage your portfolio:\n"
            "/portfolio <coin1> <percentage1> <coin2> <percentage2> ...\n"
            "Example: /portfolio BTC 50 ETH 30 SOL 20"
        )
    elif query.data == 'pnl':
        await handle_pnl(update, context)
    elif query.data == 'learn':
        await query.edit_message_text(
            "برای یادگیری، از دستور زیر استفاده کنید:\n"
            "Use the following command to learn:\n"
            "/learn <topic>\n"
            "Topics: basics, trading, security, blockchain, defi\n"
            "Example: /learn basics"
        )
    elif query.data == 'scam':
        await query.edit_message_text(
            "برای بررسی کلاهبرداری، از دستور زیر استفاده کنید:\n"
            "Use the following command to check for scams:\n"
            "/scam <project_name>\n"
            "Example: /scam ProjectX"
        )
    elif query.data == 'premium':
        await handle_premium(update, context)
    elif query.data.startswith('check_payment_'):
        await check_payment_status(update, context)

async def handle_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle news command and callback"""
    query = update.callback_query
    user_id = query.from_user.id if query else update.effective_user.id
    
    user = db.get_user(user_id)
    if not user[3]:  # If not premium
        if user[7] >= 1:  # Free plan limit
            await query.answer("Daily news limit reached. Upgrade to premium for more!")
            return
    
    news = get_crypto_news()
    if not news:
        await query.answer("Error fetching news. Please try again later.")
        return
    
    db.update_daily_counts(user_id, 'news')
    
    for article in news[:1]:  # Send only one article for free users
        message = f"📰 {article['title']}\n\n{article['summary']}\n\nSource: {article['source']}\n{article['url']}"
        
        if article['image']:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=article['image'],
                caption=message
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=message
            )
        
        analysis = analyze_with_gemini(
            f"Analyze this crypto news: {article['title']}\n{article['summary']}",
            user[2]  # User's language
        )
        if analysis:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"🤖 Analysis:\n{analysis}"
            )

async def handle_prices(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle prices command and callback"""
    query = update.callback_query
    user_id = query.from_user.id if query else update.effective_user.id
    
    prices = get_crypto_prices()
    if not prices:
        await query.answer("Error fetching prices. Please try again later.")
        return
    
    user = db.get_user(user_id)
    lang = user[2]
    
    message = "💰 Top 15 Cryptocurrencies:\n\n"
    for coin in prices:
        message += f"{coin['name']} ({coin['symbol']})\n"
        message += f"Price: {format_large_number(coin['quotes']['USD']['price'])}\n"
        message += f"24h Change: {format_price_change(coin['quotes']['USD']['percent_change_24h'])}\n"
        message += f"Market Cap: {format_large_number(coin['quotes']['USD']['market_cap'])}\n\n"
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=message
    )

async def handle_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle chart command"""
    user_id = update.effective_user.id
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: /chart <symbol> <interval>\n"
            "Example: /chart BTCUSDT 1h"
        )
        return
    
    symbol, interval = args
    
    user = db.get_user(user_id)
    if not user[3]:  # If not premium
        if interval not in ['1h']:  # Free plan limit
            await update.message.reply_text(
                "This timeframe is only available for premium users. "
                "Upgrade to access all timeframes!"
            )
            return
    
    data = get_binance_chart(symbol, interval)
    if not data:
        await update.message.reply_text("Error fetching chart data. Please try again later.")
        return
    
    chart_path = create_candlestick_chart(data, symbol, interval)
    if not chart_path:
        await update.message.reply_text("Error creating chart. Please try again later.")
        return
    
    await context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=open(chart_path, 'rb'),
        caption=f"📊 {symbol} Price Chart ({interval})"
    )
    
    os.remove(chart_path)

async def handle_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle alert command"""
    user_id = update.effective_user.id
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: /alert <symbol> <price>\n"
            "Example: /alert BTC 70000"
        )
        return
    
    symbol, price = args
    try:
        price = float(price)
    except ValueError:
        await update.message.reply_text("Invalid price. Please enter a valid number.")
        return
    
    user = db.get_user(user_id)
    if not user[3]:  # If not premium
        if user[9] >= 1:  # Free plan limit
            await update.message.reply_text(
                "Daily alert limit reached. Upgrade to premium for more alerts!"
            )
            return
    
    alert_id = db.add_price_alert(user_id, symbol, price)
    if alert_id:
        db.update_daily_counts(user_id, 'alert')
        await update.message.reply_text(
            f"✅ Alert set for {symbol} at ${price:,.2f}\n"
            "You will be notified when the price reaches this level."
        )
    else:
        await update.message.reply_text("Error setting alert. Please try again later.")

async def handle_premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle premium command and callback"""
    query = update.callback_query
    user_id = query.from_user.id if query else update.effective_user.id
    
    user = db.get_user(user_id)
    lang = user[2]
    
    message = (
        f"💎 ویژگی‌های پرمیوم:\n\n"
        f"• 5 مقاله خبری در روز با تحلیل\n"
        f"• دسترسی به تمام بازه‌های زمانی نمودار (1m, 1h, 4h, 1d)\n"
        f"• 5 هشدار قیمت\n"
        f"• 5 تحلیل روزانه\n"
        f"• پیشنهادات پورتفولیو\n"
        f"• تقویم رویدادهای کریپتو\n"
        f"• ابزار مقایسه ارزها\n\n"
        f"قیمت: ${PREMIUM_PLAN_PRICE}/ماه\n\n"
        f"آیا می‌خواهید ارتقا دهید؟"
    )
    
    keyboard = [
        [InlineKeyboardButton("💳 پرداخت با ارز دیجیتال", callback_data='crypto_payment')],
        [InlineKeyboardButton("❌ انصراف", callback_data='cancel')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if query:
        await query.edit_message_text(message, reply_markup=reply_markup)
    else:
        await update.message.reply_text(message, reply_markup=reply_markup)

async def handle_donate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle donate command"""
    user = db.get_user(update.effective_user.id)
    lang = user[2]
    
    donation_messages = {
        'fa': "ما با عشق و تلاش، این ربات رو برای شما ساختیم تا بهترین ابزار کریپتویی رو داشته باشید! 🚀 با حمایت مالی شما، می‌تونیم قابلیت‌های جدید اضافه کنیم، سرورها رو ارتقا بدیم، و تجربه بهتری براتون بسازیم. حتی یه کمک کوچیک، ما رو به هدفمون نزدیک‌تر می‌کنه. از شما ممنونیم! ❤️ آدرس تتر (USDT) روی شبکه ترون: TEboRphXkDfcD2azhyBv2VUsPqxujnjPGQ",
        'en': "We built this bot with love and dedication to give you the best crypto tool! 🚀 Your financial support helps us add new features, upgrade servers, and create a better experience for you. Even a small donation brings us closer to our goal. Thank you! ❤️ USDT (TRON) Address: TEboRphXkDfcD2azhyBv2VUsPqxujnjPGQ",
        'ar': "لقد بنينا هذا البوت بحب وتفانٍ لنقدم لك أفضل أداة للعملات الرقمية! 🚀 دعمك المالي يساعدنا على إضافة ميزات جديدة، تحسين الخوادم، وخلق تجربة أفضل لك. حتى التبرع الصغير يقربنا من هدفنا. شكرًا لك! ❤️ عنوان تتر (USDT) على شبكة ترون: TEboRphXkDfcD2azhyBv2VUsPqxujnjPGQ",
        'tr': "Bu botu size en iyi kripto aracını sunmak için sevgi ve özveriyle geliştirdik! 🚀 Finansal desteğinizle yeni özellikler ekleyebilir, sunucuları yükseltebilir ve sizin için daha iyi bir deneyim yaratabiliriz. Küçük bir bağış bile bizi hedefimize yaklaştırır. Teşekkür ederiz! ❤️ USDT (TRON) Adresi: TEboRphXkDfcD2azhyBv2VUsPqxujnjPGQ",
        'es': "¡Construimos este bot con amor y dedicación para darte la mejor herramienta cripto! 🚀 Tu apoyo financiero nos ayuda a agregar nuevas funciones, mejorar los servidores y crear una mejor experiencia para ti. Incluso una pequeña donación nos acerca a nuestra meta. ¡Gracias! ❤️ Dirección USDT (TRON): TEboRphXkDfcD2azhyBv2VUsPqxujnjPGQ"
    }
    
    await update.message.reply_text(donation_messages[lang])

async def handle_sentiment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle sentiment analysis command"""
    user_id = update.effective_user.id
    args = context.args
    
    if not args:
        await update.message.reply_text(
            "Usage: /sentiment <symbol>\n"
            "Example: /sentiment BTC"
        )
        return
    
    symbol = args[0].upper()
    user = db.get_user(user_id)
    
    # Get latest sentiment from database
    sentiment = db.get_latest_sentiment(symbol)
    if sentiment:
        await update.message.reply_text(
            f"🤖 Market Sentiment Analysis for {symbol}:\n\n"
            f"Score: {sentiment[2]:.2f}/10\n"
            f"Analysis: {sentiment[3]}"
        )
        return
    
    # Get news and prices for analysis
    news = get_crypto_news()
    prices = get_crypto_prices()
    
    if not news or not prices:
        await update.message.reply_text("Error fetching data. Please try again later.")
        return
    
    # Prepare data for analysis
    news_text = "\n".join([f"{article['title']}\n{article['summary']}" for article in news[:3]])
    price_data = next((coin for coin in prices if coin['symbol'] == symbol), None)
    
    if not price_data:
        await update.message.reply_text("Invalid symbol. Please try again.")
        return
    
    # Analyze with Gemini
    prompt = f"""
    Analyze the market sentiment for {symbol} based on:
    1. Latest news: {news_text}
    2. Price data: Current price ${price_data['quotes']['USD']['price']:.2f}, 
       24h change {price_data['quotes']['USD']['percent_change_24h']:.2f}%
    
    Provide:
    1. Sentiment score (0-10)
    2. Brief analysis
    3. Key factors affecting sentiment
    """
    
    analysis = analyze_with_gemini(prompt, user[2])
    if not analysis:
        await update.message.reply_text("Error analyzing sentiment. Please try again later.")
        return
    
    # Calculate sentiment score (simple implementation)
    sentiment_score = 5.0  # Default neutral score
    if "bullish" in analysis.lower() or "positive" in analysis.lower():
        sentiment_score += 2
    if "bearish" in analysis.lower() or "negative" in analysis.lower():
        sentiment_score -= 2
    
    # Save to database
    db.save_market_sentiment(symbol, sentiment_score, analysis)
    
    await update.message.reply_text(
        f"🤖 Market Sentiment Analysis for {symbol}:\n\n"
        f"Score: {sentiment_score:.2f}/10\n"
        f"Analysis: {analysis}"
    )

async def handle_predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle price prediction command"""
    user_id = update.effective_user.id
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: /predict <symbol> <days>\n"
            "Example: /predict BTC 7"
        )
        return
    
    symbol, days = args
    try:
        days = int(days)
        if days < 1 or days > 30:
            raise ValueError
    except ValueError:
        await update.message.reply_text("Invalid number of days. Please enter a number between 1 and 30.")
        return
    
    user = db.get_user(user_id)
    if not user[3]:  # If not premium
        await update.message.reply_text("Price prediction is a premium feature. Upgrade to access it!")
        return
    
    # Get historical data
    data = get_binance_chart(symbol, '1d', limit=30)
    if not data:
        await update.message.reply_text("Error fetching historical data. Please try again later.")
        return
    
    # Prepare data for analysis
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Analyze with Gemini
    prompt = f"""
    Analyze the price data for {symbol} and predict its price for the next {days} days.
    
    Historical data:
    - Current price: ${float(df['close'].iloc[-1]):.2f}
    - 30-day high: ${float(df['high'].max()):.2f}
    - 30-day low: ${float(df['low'].min()):.2f}
    - 30-day volume: ${float(df['quote_volume'].sum()):.2f}
    
    Provide:
    1. Predicted price range
    2. Confidence level
    3. Key factors affecting the prediction
    4. Potential risks
    """
    
    analysis = analyze_with_gemini(prompt, user[2])
    if not analysis:
        await update.message.reply_text("Error generating prediction. Please try again later.")
        return
    
    await update.message.reply_text(
        f"🤖 Price Prediction for {symbol} ({days} days):\n\n{analysis}"
    )

async def handle_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle portfolio command"""
    user_id = update.effective_user.id
    args = context.args
    
    if not args:
        await update.message.reply_text(
            "Usage: /portfolio <coin1> <percentage1> <coin2> <percentage2> ...\n"
            "Example: /portfolio BTC 50 ETH 30 SOL 20"
        )
        return
    
    user = db.get_user(user_id)
    if not user[3]:  # If not premium
        await update.message.reply_text("Portfolio analysis is a premium feature. Upgrade to access it!")
        return
    
    try:
        portfolio = {}
        total_percentage = 0
        
        for i in range(0, len(args), 2):
            if i + 1 >= len(args):
                raise ValueError
            
            coin = args[i].upper()
            percentage = float(args[i + 1])
            
            if percentage < 0 or percentage > 100:
                raise ValueError
            
            portfolio[coin] = percentage
            total_percentage += percentage
        
        if total_percentage != 100:
            raise ValueError
        
        # Save portfolio
        db.save_portfolio(user_id, portfolio)
        
        # Get current prices
        prices = get_crypto_prices()
        if not prices:
            await update.message.reply_text("Error fetching prices. Please try again later.")
            return
        
        # Prepare data for analysis
        price_data = {coin['symbol']: coin['quotes']['USD']['price'] for coin in prices}
        
        # Analyze with Gemini
        prompt = f"""
        Analyze this crypto portfolio:
        {json.dumps(portfolio, indent=2)}
        
        Current prices:
        {json.dumps(price_data, indent=2)}
        
        Provide:
        1. Portfolio diversification analysis
        2. Risk assessment
        3. Suggested adjustments
        4. Market outlook
        """
        
        analysis = analyze_with_gemini(prompt, user[2])
        if not analysis:
            await update.message.reply_text("Error analyzing portfolio. Please try again later.")
            return
        
        await update.message.reply_text(
            f"🤖 Portfolio Analysis:\n\n{analysis}"
        )
        
    except ValueError:
        await update.message.reply_text(
            "Invalid portfolio data. Please provide valid coins and percentages that sum to 100."
        )

async def handle_scam_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle scam check command"""
    user_id = update.effective_user.id
    args = context.args
    
    if not args:
        await update.message.reply_text(
            "Usage: /scam <project_name>\n"
            "Example: /scam ProjectX"
        )
        return
    
    project_name = " ".join(args)
    user = db.get_user(user_id)
    
    # Analyze with Gemini
    prompt = f"""
    Analyze this crypto project for potential scam indicators:
    Project Name: {project_name}
    
    Check for:
    1. Red flags and warning signs
    2. Team transparency
    3. Project legitimacy
    4. Investment risks
    5. Recommendations
    
    Provide a detailed analysis with a risk score (0-10).
    """
    
    analysis = analyze_with_gemini(prompt, user[2])
    if not analysis:
        await update.message.reply_text("Error analyzing project. Please try again later.")
        return
    
    await update.message.reply_text(
        f"🔍 Scam Analysis for {project_name}:\n\n{analysis}"
    )

async def handle_technical(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle technical analysis command"""
    user_id = update.effective_user.id
    args = context.args
    
    if not args:
        await update.message.reply_text(
            "Usage: /technical <symbol> <timeframe>\n"
            "Example: /technical BTC 1d"
        )
        return
    
    symbol, timeframe = args
    user = db.get_user(user_id)
    
    # Get historical data
    data = get_binance_chart(symbol, timeframe, limit=100)
    if not data:
        await update.message.reply_text("Error fetching chart data. Please try again later.")
        return
    
    # Prepare data for analysis
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate technical indicators
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['close'])
    
    # Analyze with Gemini
    prompt = f"""
    Analyze technical patterns for {symbol} on {timeframe} timeframe:
    
    Current data:
    - Price: ${float(df['close'].iloc[-1]):.2f}
    - 20 SMA: ${float(df['SMA_20'].iloc[-1]):.2f}
    - 50 SMA: ${float(df['SMA_50'].iloc[-1]):.2f}
    - RSI: {float(df['RSI'].iloc[-1]):.2f}
    
    Identify:
    1. Chart patterns (support/resistance, trend lines)
    2. Technical indicators signals
    3. Potential entry/exit points
    4. Risk levels
    5. Trading recommendations
    """
    
    analysis = analyze_with_gemini(prompt, user[2])
    if not analysis:
        await update.message.reply_text("Error analyzing technical patterns. Please try again later.")
        return
    
    await update.message.reply_text(
        f"📊 Technical Analysis for {symbol} ({timeframe}):\n\n{analysis}"
    )

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

async def handle_compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle coin comparison command"""
    user_id = update.effective_user.id
    args = context.args
    
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: /compare <coin1> <coin2>\n"
            "Example: /compare BTC ETH"
        )
        return
    
    coin1, coin2 = args
    user = db.get_user(user_id)
    
    # Get prices
    prices = get_crypto_prices()
    if not prices:
        await update.message.reply_text("Error fetching prices. Please try again later.")
        return
    
    # Find coin data
    coin1_data = next((coin for coin in prices if coin['symbol'].upper() == coin1.upper()), None)
    coin2_data = next((coin for coin in prices if coin['symbol'].upper() == coin2.upper()), None)
    
    if not coin1_data or not coin2_data:
        await update.message.reply_text("Invalid coin symbols. Please try again.")
        return
    
    # Analyze with Gemini
    prompt = f"""
    Compare these cryptocurrencies:
    
    {coin1_data['name']} ({coin1_data['symbol']}):
    - Price: ${coin1_data['quotes']['USD']['price']:.2f}
    - 24h Change: {coin1_data['quotes']['USD']['percent_change_24h']:.2f}%
    - Market Cap: ${coin1_data['quotes']['USD']['market_cap']:.2f}
    - Volume: ${coin1_data['quotes']['USD']['volume_24h']:.2f}
    
    {coin2_data['name']} ({coin2_data['symbol']}):
    - Price: ${coin2_data['quotes']['USD']['price']:.2f}
    - 24h Change: {coin2_data['quotes']['USD']['percent_change_24h']:.2f}%
    - Market Cap: ${coin2_data['quotes']['USD']['market_cap']:.2f}
    - Volume: ${coin2_data['quotes']['USD']['volume_24h']:.2f}
    
    Provide:
    1. Performance comparison
    2. Market position analysis
    3. Risk assessment
    4. Investment potential
    5. Key differences
    """
    
    analysis = analyze_with_gemini(prompt, user[2])
    if not analysis:
        await update.message.reply_text("Error comparing coins. Please try again later.")
        return
    
    await update.message.reply_text(
        f"⚖️ Comparison: {coin1_data['name']} vs {coin2_data['name']}\n\n{analysis}"
    )

async def handle_news_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle news alert command"""
    user_id = update.effective_user.id
    args = context.args
    
    if not args:
        await update.message.reply_text(
            "Usage: /newsalert <keyword>\n"
            "Example: /newsalert Bitcoin"
        )
        return
    
    keyword = " ".join(args)
    user = db.get_user(user_id)
    
    # Get news
    news = get_crypto_news()
    if not news:
        await update.message.reply_text("Error fetching news. Please try again later.")
        return
    
    # Filter news by keyword
    relevant_news = [
        article for article in news
        if keyword.lower() in article['title'].lower() or
        keyword.lower() in article['summary'].lower()
    ]
    
    if not relevant_news:
        await update.message.reply_text(f"No news found for '{keyword}'. Try a different keyword.")
        return
    
    # Send relevant news
    for article in relevant_news[:3]:  # Limit to 3 articles
        message = f"🔔 News Alert for '{keyword}':\n\n"
        message += f"📰 {article['title']}\n\n"
        message += f"{article['summary']}\n\n"
        message += f"Source: {article['source']}\n"
        message += f"{article['url']}"
        
        if article['image']:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=article['image'],
                caption=message
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=message
            )

async def handle_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle portfolio P&L calculation command"""
    user_id = update.effective_user.id
    user = db.get_user(user_id)
    
    # Get portfolio
    portfolio = db.get_portfolio(user_id)
    if not portfolio:
        await update.message.reply_text(
            "No portfolio found. Use /portfolio command to create one."
        )
        return
    
    # Get current prices
    prices = get_crypto_prices()
    if not prices:
        await update.message.reply_text("Error fetching prices. Please try again later.")
        return
    
    # Calculate P&L
    price_data = {coin['symbol']: coin['quotes']['USD']['price'] for coin in prices}
    total_value = 0
    pnl_data = {}
    
    for coin, percentage in portfolio.items():
        if coin in price_data:
            current_price = price_data[coin]
            # Assuming initial investment of $1000 for calculation
            initial_value = 1000 * (percentage / 100)
            current_value = initial_value * (current_price / price_data[coin])
            pnl = current_value - initial_value
            pnl_percentage = (pnl / initial_value) * 100
            
            pnl_data[coin] = {
                'initial_value': initial_value,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage
            }
            total_value += current_value
    
    # Analyze with Gemini
    prompt = f"""
    Analyze this portfolio P&L:
    {json.dumps(pnl_data, indent=2)}
    
    Total Portfolio Value: ${total_value:.2f}
    
    Provide:
    1. Overall performance analysis
    2. Best and worst performing assets
    3. Risk assessment
    4. Portfolio optimization suggestions
    5. Market outlook impact
    """
    
    analysis = analyze_with_gemini(prompt, user[2])
    if not analysis:
        await update.message.reply_text("Error analyzing P&L. Please try again later.")
        return
    
    # Format P&L message
    message = f"💰 Portfolio P&L Analysis:\n\n"
    for coin, data in pnl_data.items():
        message += f"{coin}:\n"
        message += f"Initial Value: ${data['initial_value']:.2f}\n"
        message += f"Current Value: ${data['current_value']:.2f}\n"
        message += f"P&L: ${data['pnl']:.2f} ({data['pnl_percentage']:.2f}%)\n\n"
    
    message += f"Total Portfolio Value: ${total_value:.2f}\n\n"
    message += f"Analysis:\n{analysis}"
    
    await update.message.reply_text(message)

async def handle_learn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle crypto education command"""
    user_id = update.effective_user.id
    args = context.args
    
    if not args:
        await update.message.reply_text(
            "Usage: /learn <topic>\n"
            "Topics: basics, trading, security, blockchain, defi\n"
            "Example: /learn basics"
        )
        return
    
    topic = args[0].lower()
    user = db.get_user(user_id)
    
    topics = {
        'basics': 'Cryptocurrency basics and fundamentals',
        'trading': 'Trading strategies and analysis',
        'security': 'Crypto security best practices',
        'blockchain': 'Blockchain technology explained',
        'defi': 'Decentralized Finance (DeFi) overview'
    }
    
    if topic not in topics:
        await update.message.reply_text(
            "Invalid topic. Available topics: basics, trading, security, blockchain, defi"
        )
        return
    
    # Analyze with Gemini
    prompt = f"""
    Create a comprehensive educational guide about {topics[topic]}.
    
    Include:
    1. Basic concepts and definitions
    2. Key components and features
    3. How it works
    4. Benefits and risks
    5. Practical examples
    6. Tips for beginners
    7. Common mistakes to avoid
    8. Resources for further learning
    
    Make it easy to understand for beginners.
    """
    
    analysis = analyze_with_gemini(prompt, user[2])
    if not analysis:
        await update.message.reply_text("Error generating educational content. Please try again later.")
        return
    
    await update.message.reply_text(
        f"📚 Educational Guide: {topics[topic]}\n\n{analysis}"
    )

async def handle_payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle cryptocurrency payment"""
    user_id = update.effective_user.id
    user = db.get_user(user_id)
    
    # Create payment invoice
    headers = {
        'x-api-key': NOWPAYMENTS_API_KEY,
        'Content-Type': 'application/json'
    }
    
    data = {
        'price_amount': PREMIUM_PLAN_PRICE,
        'price_currency': 'usd',
        'order_id': f'premium_{user_id}_{int(datetime.now().timestamp())}',
        'order_description': 'Premium Plan Subscription',
        'ipn_callback_url': 'https://your-domain.com/ipn',  # Replace with your domain
        'success_url': 'https://t.me/your_bot_username',  # Replace with your bot username
        'cancel_url': 'https://t.me/your_bot_username'  # Replace with your bot username
    }
    
    try:
        response = requests.post(
            NOWPAYMENTS_API_URL,
            headers=headers,
            json=data
        )
        response.raise_for_status()
        payment_data = response.json()
        
        # Save payment info to database
        db.add_payment(
            user_id=user_id,
            order_id=payment_data['order_id'],
            amount=PREMIUM_PLAN_PRICE,
            currency='usd',
            status='pending'
        )
        
        # Create payment message
        message = (
            f"💳 پرداخت پرمیوم\n\n"
            f"مبلغ: ${PREMIUM_PLAN_PRICE}\n"
            f"شناسه سفارش: {payment_data['order_id']}\n\n"
            f"لطفاً از طریق لینک زیر پرداخت کنید:\n"
            f"{payment_data['invoice_url']}\n\n"
            f"پس از پرداخت موفق، دسترسی پرمیوم شما فعال خواهد شد."
        )
        
        # Add payment status check button
        keyboard = [[InlineKeyboardButton("🔄 بررسی وضعیت پرداخت", callback_data=f'check_payment_{payment_data["order_id"]}')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Payment error: {e}")
        await update.message.reply_text(
            "خطا در ایجاد پرداخت. لطفاً دوباره تلاش کنید."
        )

async def check_payment_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check payment status"""
    query = update.callback_query
    await query.answer()
    
    order_id = query.data.split('_')[2]
    user_id = query.from_user.id
    
    # Get payment info from database
    payment = db.get_payment(order_id)
    if not payment:
        await query.edit_message_text("اطلاعات پرداخت یافت نشد.")
        return
    
    # Check payment status with NOWPayments
    headers = {
        'x-api-key': NOWPAYMENTS_API_KEY
    }
    
    try:
        response = requests.get(
            f"{NOWPAYMENTS_API_URL}/{order_id}",
            headers=headers
        )
        response.raise_for_status()
        payment_status = response.json()
        
        if payment_status['payment_status'] == 'finished':
            # Update user premium status
            expiry_date = datetime.now() + timedelta(days=30)
            db.update_user_premium(user_id, True, expiry_date)
            
            # Update payment status
            db.add_payment(
                user_id=user_id,
                order_id=order_id,
                amount=payment[3],
                currency=payment[4],
                status='completed'
            )
            
            await query.edit_message_text(
                "✅ پرداخت با موفقیت انجام شد!\n"
                "دسترسی پرمیوم شما فعال شد."
            )
        else:
            await query.edit_message_text(
                "⏳ پرداخت در حال پردازش است.\n"
                "لطفاً چند دقیقه دیگر دوباره بررسی کنید."
            )
            
    except Exception as e:
        logger.error(f"Payment status check error: {e}")
        await query.edit_message_text(
            "خطا در بررسی وضعیت پرداخت. لطفاً دوباره تلاش کنید."
        )

def main():
    """Start the bot"""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("news", handle_news))
    application.add_handler(CommandHandler("prices", handle_prices))
    application.add_handler(CommandHandler("chart", handle_chart))
    application.add_handler(CommandHandler("alert", handle_alert))
    application.add_handler(CommandHandler("premium", handle_premium))
    application.add_handler(CommandHandler("donate", handle_donate))
    application.add_handler(CommandHandler("sentiment", handle_sentiment))
    application.add_handler(CommandHandler("predict", handle_predict))
    application.add_handler(CommandHandler("portfolio", handle_portfolio))
    application.add_handler(CommandHandler("scam", handle_scam_check))
    application.add_handler(CommandHandler("technical", handle_technical))
    application.add_handler(CommandHandler("compare", handle_compare))
    application.add_handler(CommandHandler("newsalert", handle_news_alert))
    application.add_handler(CommandHandler("pnl", handle_pnl))
    application.add_handler(CommandHandler("learn", handle_learn))
    application.add_handler(CommandHandler("payment", handle_payment))
    
    # Add callback handlers
    application.add_handler(CallbackQueryHandler(language_callback, pattern='^lang_'))
    application.add_handler(CallbackQueryHandler(handle_callback))
    
    # Start the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 
