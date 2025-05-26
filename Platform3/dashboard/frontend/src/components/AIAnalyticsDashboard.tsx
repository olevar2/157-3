import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  Chip,
  Button,
  TextField,
  IconButton,
  Avatar,
  Divider,
  Tab,
  Tabs,
  Alert,
  LinearProgress
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Chat,
  Send,
  SmartToy,
  Assessment,
  Recommendations,
  Timeline,
  AttachMoney,
  Warning,
  CheckCircle
} from '@mui/icons-material';
import { useWebSocket } from '../contexts/WebSocketContext';

interface PairAnalysis {
  pair: string;
  trend: 'bullish' | 'bearish' | 'neutral';
  strength: number;
  recommendation: 'buy' | 'sell' | 'hold';
  confidence: number;
  targetPrice: number;
  stopLoss: number;
  analysis: string;
  technicalIndicators: {
    rsi: number;
    macd: string;
    sma20: number;
    sma50: number;
    support: number;
    resistance: number;
  };
}

interface ChatMessage {
  id: string;
  type: 'user' | 'ai';
  message: string;
  timestamp: Date;
  analysis?: PairAnalysis;
  recommendations?: string[];
}

interface TradingSignal {
  id: string;
  pair: string;
  action: 'buy' | 'sell';
  price: number;
  confidence: number;
  reasoning: string;
  timestamp: Date;
  status: 'pending' | 'executed' | 'expired';
}

const AIAnalyticsDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [pairAnalyses, setPairAnalyses] = useState<PairAnalysis[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [tradingSignals, setTradingSignals] = useState<TradingSignal[]>([]);
  const [isAIThinking, setIsAIThinking] = useState(false);
  const { sendMessage, lastMessage } = useWebSocket();

  // Mock data for demonstration
  useEffect(() => {
    // Initialize with sample data
    setPairAnalyses([
      {
        pair: 'EUR/USD',
        trend: 'bullish',
        strength: 75,
        recommendation: 'buy',
        confidence: 85,
        targetPrice: 1.0850,
        stopLoss: 1.0720,
        analysis: 'Strong bullish momentum confirmed by multiple technical indicators. ECB dovish stance and Fed policy uncertainty creating favorable conditions.',
        technicalIndicators: {
          rsi: 65,
          macd: 'bullish_crossover',
          sma20: 1.0785,
          sma50: 1.0745,
          support: 1.0720,
          resistance: 1.0850
        }
      },
      {
        pair: 'GBP/USD',
        trend: 'bearish',
        strength: 60,
        recommendation: 'sell',
        confidence: 70,
        targetPrice: 1.2450,
        stopLoss: 1.2580,
        analysis: 'Bearish pressure from UK economic data. BoE policy uncertainty and Brexit concerns weighing on GBP.',
        technicalIndicators: {
          rsi: 35,
          macd: 'bearish_divergence',
          sma20: 1.2520,
          sma50: 1.2565,
          support: 1.2450,
          resistance: 1.2580
        }
      },
      {
        pair: 'USD/JPY',
        trend: 'neutral',
        strength: 45,
        recommendation: 'hold',
        confidence: 55,
        targetPrice: 150.25,
        stopLoss: 148.80,
        analysis: 'Consolidation phase with mixed signals. Awaiting clearer directional bias from economic data.',
        technicalIndicators: {
          rsi: 52,
          macd: 'neutral',
          sma20: 149.85,
          sma50: 149.70,
          support: 148.80,
          resistance: 150.60
        }
      }
    ]);

    setTradingSignals([
      {
        id: '1',
        pair: 'EUR/USD',
        action: 'buy',
        price: 1.0780,
        confidence: 85,
        reasoning: 'Technical breakout above resistance with strong volume confirmation',
        timestamp: new Date(),
        status: 'pending'
      },
      {
        id: '2',
        pair: 'GBP/USD',
        action: 'sell',
        price: 1.2520,
        confidence: 70,
        reasoning: 'Bearish divergence on RSI with fundamental weakness',
        timestamp: new Date(Date.now() - 1800000),
        status: 'executed'
      }
    ]);

    setChatMessages([
      {
        id: '1',
        type: 'ai',
        message: 'Hello! I\'m your AI trading assistant. I\'ve analyzed the current market conditions and have several recommendations for you. How can I help you today?',
        timestamp: new Date(),
        recommendations: [
          'View EUR/USD bullish setup',
          'Check current market sentiment',
          'Review portfolio performance',
          'Get risk management advice'
        ]
      }
    ]);
  }, []);

  const handleSendMessage = async () => {
    if (!currentMessage.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      message: currentMessage,
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsAIThinking(true);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        message: generateAIResponse(currentMessage),
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, aiResponse]);
      setIsAIThinking(false);
    }, 2000);
  };

  const generateAIResponse = (userMessage: string): string => {
    const lowerMessage = userMessage.toLowerCase();
    
    if (lowerMessage.includes('eur') || lowerMessage.includes('euro')) {
      return 'Based on my analysis, EUR/USD is showing strong bullish momentum. The pair has broken above key resistance at 1.0760 with good volume. I recommend a buy position with target at 1.0850 and stop loss at 1.0720. The ECB\'s dovish stance and improving eurozone data support this view.';
    } else if (lowerMessage.includes('gbp') || lowerMessage.includes('pound')) {
      return 'GBP/USD is currently under bearish pressure. UK economic uncertainty and BoE policy confusion are weighing on the pound. I suggest caution and potentially a sell bias with targets around 1.2450. Watch for any Brexit-related news that could cause volatility.';
    } else if (lowerMessage.includes('market') || lowerMessage.includes('sentiment')) {
      return 'Current market sentiment is mixed with selective strength in EUR and weakness in GBP. USD is consolidating after recent moves. I\'m monitoring central bank communications and economic data for directional clues. Risk-on sentiment is gradually improving.';
    } else if (lowerMessage.includes('recommend') || lowerMessage.includes('suggestion')) {
      return 'My top recommendations right now: 1) EUR/USD long position (85% confidence), 2) GBP/USD short position (70% confidence), 3) Monitor USD/JPY for breakout. I suggest position sizing at 2% risk per trade with tight stop losses given current volatility.';
    } else {
      return 'I understand you\'re looking for trading insights. Based on my real-time analysis of market data, technical indicators, and fundamental factors, I can provide specific recommendations for any currency pair. What specific aspect of the market would you like me to analyze for you?';
    }
  };

  const executeSignal = (signalId: string) => {
    setTradingSignals(prev => 
      prev.map(signal => 
        signal.id === signalId 
          ? { ...signal, status: 'executed' as const }
          : signal
      )
    );
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'bullish': return <TrendingUp color="success" />;
      case 'bearish': return <TrendingDown color="error" />;
      default: return <Timeline color="warning" />;
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'bullish': return 'success';
      case 'bearish': return 'error';
      default: return 'warning';
    }
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        <SmartToy sx={{ mr: 2, verticalAlign: 'middle' }} />
        AI Analytics & Trading Assistant
      </Typography>

      <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)} sx={{ mb: 3 }}>
        <Tab label="Market Analysis" icon={<Assessment />} />
        <Tab label="AI Chat" icon={<Chat />} />
        <Tab label="Trading Signals" icon={<Recommendations />} />
      </Tabs>

      {/* Market Analysis Tab */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Real-Time Pair Analysis & Recommendations
            </Typography>
          </Grid>
          
          {pairAnalyses.map((analysis, index) => (
            <Grid item xs={12} md={6} lg={4} key={index}>
              <Card elevation={3}>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h6">{analysis.pair}</Typography>
                    {getTrendIcon(analysis.trend)}
                  </Box>
                  
                  <Chip 
                    label={analysis.recommendation.toUpperCase()} 
                    color={analysis.recommendation === 'buy' ? 'success' : analysis.recommendation === 'sell' ? 'error' : 'default'}
                    sx={{ mb: 2 }}
                  />
                  
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {analysis.analysis}
                  </Typography>
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      Confidence: {analysis.confidence}%
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={analysis.confidence} 
                      color={getTrendColor(analysis.trend)}
                      sx={{ mt: 1 }}
                    />
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="subtitle2" gutterBottom>
                    Key Levels:
                  </Typography>
                  <Typography variant="body2">
                    Target: {analysis.targetPrice}
                  </Typography>
                  <Typography variant="body2">
                    Stop Loss: {analysis.stopLoss}
                  </Typography>
                  <Typography variant="body2">
                    Support: {analysis.technicalIndicators.support}
                  </Typography>
                  <Typography variant="body2">
                    Resistance: {analysis.technicalIndicators.resistance}
                  </Typography>
                  
                  <Button 
                    fullWidth 
                    variant="contained" 
                    sx={{ mt: 2 }}
                    color={analysis.recommendation === 'buy' ? 'success' : analysis.recommendation === 'sell' ? 'error' : 'primary'}
                  >
                    Execute {analysis.recommendation.toUpperCase()}
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* AI Chat Tab */}
      {activeTab === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Paper elevation={3} sx={{ height: 600, display: 'flex', flexDirection: 'column' }}>
              <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                <Typography variant="h6">
                  <SmartToy sx={{ mr: 1, verticalAlign: 'middle' }} />
                  AI Trading Assistant
                </Typography>
              </Box>
              
              <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
                {chatMessages.map((message) => (
                  <Box
                    key={message.id}
                    sx={{
                      display: 'flex',
                      justifyContent: message.type === 'user' ? 'flex-end' : 'flex-start',
                      mb: 2
                    }}
                  >
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        maxWidth: '70%',
                        flexDirection: message.type === 'user' ? 'row-reverse' : 'row'
                      }}
                    >
                      <Avatar
                        sx={{
                          bgcolor: message.type === 'user' ? 'primary.main' : 'secondary.main',
                          mx: 1
                        }}
                      >
                        {message.type === 'user' ? 'U' : <SmartToy />}
                      </Avatar>
                      <Paper
                        elevation={1}
                        sx={{
                          p: 2,
                          bgcolor: message.type === 'user' ? 'primary.light' : 'grey.100',
                          color: message.type === 'user' ? 'white' : 'text.primary'
                        }}
                      >
                        <Typography variant="body1">{message.message}</Typography>
                        {message.recommendations && (
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="body2" sx={{ mb: 1, fontWeight: 'bold' }}>
                              Quick Actions:
                            </Typography>
                            {message.recommendations.map((rec, index) => (
                              <Chip
                                key={index}
                                label={rec}
                                size="small"
                                sx={{ mr: 1, mb: 1 }}
                                onClick={() => setCurrentMessage(rec)}
                              />
                            ))}
                          </Box>
                        )}
                        <Typography variant="caption" display="block" sx={{ mt: 1, opacity: 0.7 }}>
                          {message.timestamp.toLocaleTimeString()}
                        </Typography>
                      </Paper>
                    </Box>
                  </Box>
                ))}
                
                {isAIThinking && (
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Avatar sx={{ bgcolor: 'secondary.main', mr: 1 }}>
                      <SmartToy />
                    </Avatar>
                    <Paper elevation={1} sx={{ p: 2, bgcolor: 'grey.100' }}>
                      <Typography variant="body1">AI is analyzing...</Typography>
                      <LinearProgress sx={{ mt: 1 }} />
                    </Paper>
                  </Box>
                )}
              </Box>
              
              <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    fullWidth
                    variant="outlined"
                    placeholder="Ask me about market analysis, trading recommendations, or any forex questions..."
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  />
                  <IconButton 
                    color="primary" 
                    onClick={handleSendMessage}
                    disabled={!currentMessage.trim() || isAIThinking}
                  >
                    <Send />
                  </IconButton>
                </Box>
              </Box>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Paper elevation={3} sx={{ p: 2, height: 600 }}>
              <Typography variant="h6" gutterBottom>
                Quick Insights
              </Typography>
              
              <List>
                <ListItem>
                  <ListItemText
                    primary="Market Sentiment"
                    secondary="Mixed - EUR strength, GBP weakness"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="High Probability Setups"
                    secondary="EUR/USD long, GBP/USD short"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Risk Level"
                    secondary="Moderate - Watch central bank news"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Trading Session"
                    secondary="London/New York overlap - High volatility"
                  />
                </ListItem>
              </List>
              
              <Alert severity="info" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  AI is continuously monitoring 28 currency pairs and 50+ technical indicators to provide real-time recommendations.
                </Typography>
              </Alert>
            </Paper>
          </Grid>
        </Grid>
      )}

      {/* Trading Signals Tab */}
      {activeTab === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              AI-Generated Trading Signals
            </Typography>
          </Grid>
          
          {tradingSignals.map((signal) => (
            <Grid item xs={12} md={6} key={signal.id}>
              <Card elevation={3}>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h6">{signal.pair}</Typography>
                    <Chip 
                      label={signal.action.toUpperCase()} 
                      color={signal.action === 'buy' ? 'success' : 'error'}
                    />
                  </Box>
                  
                  <Typography variant="body2" paragraph>
                    {signal.reasoning}
                  </Typography>
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      Entry Price: {signal.price}
                    </Typography>
                    <Typography variant="body2">
                      Confidence: {signal.confidence}%
                    </Typography>
                    <Typography variant="body2">
                      Time: {signal.timestamp.toLocaleString()}
                    </Typography>
                  </Box>
                  
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Chip 
                      label={signal.status.toUpperCase()}
                      color={signal.status === 'executed' ? 'success' : signal.status === 'pending' ? 'warning' : 'default'}
                      icon={signal.status === 'executed' ? <CheckCircle /> : <Warning />}
                    />
                    
                    {signal.status === 'pending' && (
                      <Button 
                        variant="contained" 
                        size="small"
                        onClick={() => executeSignal(signal.id)}
                        color={signal.action === 'buy' ? 'success' : 'error'}
                      >
                        Execute Trade
                      </Button>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
};

export default AIAnalyticsDashboard;
