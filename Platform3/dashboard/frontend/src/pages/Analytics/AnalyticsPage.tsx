import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Assessment } from '@mui/icons-material';

const AnalyticsPage: React.FC = () => {
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        <Assessment sx={{ mr: 2, verticalAlign: 'middle' }} />
        Advanced Analytics
      </Typography>
      
      <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          Advanced analytics features coming soon...
        </Typography>
        <Typography variant="body1" sx={{ mt: 2 }}>
          This section will include detailed market analysis, trading statistics, and performance reports.
        </Typography>
      </Paper>
    </Box>
  );
};

export default AnalyticsPage;
