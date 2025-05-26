import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { AccountBalance } from '@mui/icons-material';

const PortfolioPage: React.FC = () => {
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        <AccountBalance sx={{ mr: 2, verticalAlign: 'middle' }} />
        Portfolio Management
      </Typography>
      
      <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          Portfolio features coming soon...
        </Typography>
        <Typography variant="body1" sx={{ mt: 2 }}>
          This section will include detailed portfolio analytics, performance tracking, and risk metrics.
        </Typography>
      </Paper>
    </Box>
  );
};

export default PortfolioPage;
