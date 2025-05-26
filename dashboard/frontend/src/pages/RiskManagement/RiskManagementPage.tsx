import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Security } from '@mui/icons-material';

const RiskManagementPage: React.FC = () => {
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        <Security sx={{ mr: 2, verticalAlign: 'middle' }} />
        Risk Management
      </Typography>
      
      <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          Risk management features coming soon...
        </Typography>
        <Typography variant="body1" sx={{ mt: 2 }}>
          This section will include risk monitoring, position sizing, and automated stop-loss management.
        </Typography>
      </Paper>
    </Box>
  );
};

export default RiskManagementPage;
