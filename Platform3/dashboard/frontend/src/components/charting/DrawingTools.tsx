/**
 * Chart Drawing Tools Component
 * Professional drawing tools for technical analysis
 * 
 * Features:
 * - Trend lines and channels
 * - Support and resistance levels
 * - Fibonacci retracements and extensions
 * - Geometric shapes and annotations
 * - Text labels and notes
 * - Measurement tools
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Tooltip,
  ButtonGroup,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Slider,
  Popover,
  Grid,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Chip
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Remove,
  Add,
  CropSquare,
  RadioButtonUnchecked,
  Timeline,
  StraightIcon,
  Edit,
  FormatColorFill,
  LineWeight,
  Delete,
  Visibility,
  VisibilityOff,
  ContentCopy,
  Save,
  Restore
} from '@mui/icons-material';

interface DrawingTool {
  id: string;
  type: 'trendline' | 'horizontal' | 'vertical' | 'rectangle' | 'circle' | 'fibonacci' | 'text' | 'arrow' | 'channel';
  name: string;
  points: Array<{ x: number; y: number; time?: number; price?: number }>;
  style: {
    color: string;
    lineWidth: number;
    lineStyle: 'solid' | 'dashed' | 'dotted';
    fillColor?: string;
    fillOpacity?: number;
  };
  text?: string;
  visible: boolean;
  locked: boolean;
  created: Date;
}

interface DrawingToolsProps {
  activeDrawings: DrawingTool[];
  selectedTool: string | null;
  onToolSelect: (tool: string | null) => void;
  onDrawingAdd: (drawing: DrawingTool) => void;
  onDrawingUpdate: (id: string, updates: Partial<DrawingTool>) => void;
  onDrawingRemove: (id: string) => void;
  onDrawingsClear: () => void;
}

const DrawingTools: React.FC<DrawingToolsProps> = ({
  activeDrawings,
  selectedTool,
  onToolSelect,
  onDrawingAdd,
  onDrawingUpdate,
  onDrawingRemove,
  onDrawingsClear
}) => {
  const [stylePopoverAnchor, setStylePopoverAnchor] = useState<HTMLElement | null>(null);
  const [currentStyle, setCurrentStyle] = useState({
    color: '#2196F3',
    lineWidth: 2,
    lineStyle: 'solid' as const,
    fillColor: '#2196F320',
    fillOpacity: 0.2
  });
  const [textInput, setTextInput] = useState('');

  // Drawing tool configurations
  const drawingTools = [
    {
      id: 'trendline',
      name: 'Trend Line',
      icon: <TrendingUp />,
      description: 'Draw trend lines to identify price direction'
    },
    {
      id: 'horizontal',
      name: 'Horizontal Line',
      icon: <Remove />,
      description: 'Draw horizontal support/resistance levels'
    },
    {
      id: 'vertical',
      name: 'Vertical Line',
      icon: <StraightIcon style={{ transform: 'rotate(90deg)' }} />,
      description: 'Draw vertical time lines'
    },
    {
      id: 'rectangle',
      name: 'Rectangle',
      icon: <CropSquare />,
      description: 'Draw rectangular areas'
    },
    {
      id: 'circle',
      name: 'Circle',
      icon: <RadioButtonUnchecked />,
      description: 'Draw circular areas'
    },
    {
      id: 'fibonacci',
      name: 'Fibonacci',
      icon: <Timeline />,
      description: 'Fibonacci retracement levels'
    },
    {
      id: 'text',
      name: 'Text',
      icon: <Edit />,
      description: 'Add text annotations'
    },
    {
      id: 'arrow',
      name: 'Arrow',
      icon: <TrendingUp style={{ transform: 'rotate(45deg)' }} />,
      description: 'Draw directional arrows'
    },
    {
      id: 'channel',
      name: 'Channel',
      icon: <Timeline />,
      description: 'Draw parallel trend channels'
    }
  ];

  // Handle tool selection
  const handleToolSelect = (toolId: string) => {
    if (selectedTool === toolId) {
      onToolSelect(null);
    } else {
      onToolSelect(toolId);
    }
  };

  // Create new drawing
  const createDrawing = useCallback((type: string, points: Array<{ x: number; y: number }>) => {
    const newDrawing: DrawingTool = {
      id: `${type}_${Date.now()}`,
      type: type as any,
      name: `${type.charAt(0).toUpperCase() + type.slice(1)} ${activeDrawings.length + 1}`,
      points,
      style: { ...currentStyle },
      text: type === 'text' ? textInput : undefined,
      visible: true,
      locked: false,
      created: new Date()
    };

    onDrawingAdd(newDrawing);
  }, [activeDrawings.length, currentStyle, textInput, onDrawingAdd]);

  // Toggle drawing visibility
  const toggleVisibility = (id: string) => {
    const drawing = activeDrawings.find(d => d.id === id);
    if (drawing) {
      onDrawingUpdate(id, { visible: !drawing.visible });
    }
  };

  // Duplicate drawing
  const duplicateDrawing = (id: string) => {
    const drawing = activeDrawings.find(d => d.id === id);
    if (drawing) {
      const duplicate: DrawingTool = {
        ...drawing,
        id: `${drawing.type}_${Date.now()}`,
        name: `${drawing.name} Copy`,
        points: drawing.points.map(p => ({ ...p, x: p.x + 10, y: p.y + 10 })),
        created: new Date()
      };
      onDrawingAdd(duplicate);
    }
  };

  // Save drawings template
  const saveTemplate = () => {
    const template = {
      name: `Template ${new Date().toLocaleDateString()}`,
      drawings: activeDrawings,
      created: new Date()
    };
    
    const savedTemplates = JSON.parse(localStorage.getItem('drawingTemplates') || '[]');
    savedTemplates.push(template);
    localStorage.setItem('drawingTemplates', JSON.stringify(savedTemplates));
  };

  // Load drawings template
  const loadTemplate = () => {
    const savedTemplates = JSON.parse(localStorage.getItem('drawingTemplates') || '[]');
    if (savedTemplates.length > 0) {
      const latestTemplate = savedTemplates[savedTemplates.length - 1];
      latestTemplate.drawings.forEach((drawing: DrawingTool) => {
        onDrawingAdd({
          ...drawing,
          id: `${drawing.type}_${Date.now()}_${Math.random()}`,
          created: new Date()
        });
      });
    }
  };

  return (
    <Box>
      {/* Drawing Tools Toolbar */}
      <Paper elevation={2} sx={{ p: 1, mb: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Drawing Tools
        </Typography>
        
        <Grid container spacing={1}>
          {drawingTools.map((tool) => (
            <Grid item key={tool.id}>
              <Tooltip title={tool.description}>
                <IconButton
                  size="small"
                  color={selectedTool === tool.id ? 'primary' : 'default'}
                  onClick={() => handleToolSelect(tool.id)}
                  sx={{
                    border: selectedTool === tool.id ? '2px solid' : '1px solid transparent',
                    borderColor: selectedTool === tool.id ? 'primary.main' : 'transparent'
                  }}
                >
                  {tool.icon}
                </IconButton>
              </Tooltip>
            </Grid>
          ))}
        </Grid>

        <Divider sx={{ my: 1 }} />

        {/* Style Controls */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
          <Tooltip title="Line Style">
            <IconButton
              size="small"
              onClick={(e) => setStylePopoverAnchor(e.currentTarget)}
            >
              <FormatColorFill />
            </IconButton>
          </Tooltip>

          <Chip
            label={`${currentStyle.lineWidth}px`}
            size="small"
            variant="outlined"
            onClick={(e) => setStylePopoverAnchor(e.currentTarget)}
          />

          <Box
            sx={{
              width: 20,
              height: 20,
              backgroundColor: currentStyle.color,
              border: '1px solid #ccc',
              borderRadius: 1,
              cursor: 'pointer'
            }}
            onClick={(e) => setStylePopoverAnchor(e.currentTarget)}
          />

          {selectedTool === 'text' && (
            <TextField
              size="small"
              placeholder="Enter text..."
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              sx={{ width: 120 }}
            />
          )}

          <Divider orientation="vertical" flexItem />

          <Tooltip title="Save Template">
            <IconButton size="small" onClick={saveTemplate}>
              <Save />
            </IconButton>
          </Tooltip>

          <Tooltip title="Load Template">
            <IconButton size="small" onClick={loadTemplate}>
              <Restore />
            </IconButton>
          </Tooltip>

          <Tooltip title="Clear All">
            <IconButton size="small" onClick={onDrawingsClear} color="error">
              <Delete />
            </IconButton>
          </Tooltip>
        </Box>
      </Paper>

      {/* Active Drawings List */}
      {activeDrawings.length > 0 && (
        <Paper elevation={2} sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Active Drawings ({activeDrawings.length})
          </Typography>
          
          <List dense>
            {activeDrawings.map((drawing) => (
              <ListItem key={drawing.id} divider>
                <ListItemIcon>
                  <Box
                    sx={{
                      width: 16,
                      height: 16,
                      backgroundColor: drawing.style.color,
                      borderRadius: drawing.type === 'circle' ? '50%' : 1
                    }}
                  />
                </ListItemIcon>
                
                <ListItemText
                  primary={drawing.name}
                  secondary={`${drawing.type} • ${drawing.created.toLocaleTimeString()}`}
                  sx={{
                    opacity: drawing.visible ? 1 : 0.5,
                    textDecoration: drawing.locked ? 'none' : 'none'
                  }}
                />
                
                <ListItemSecondaryAction>
                  <Box sx={{ display: 'flex', gap: 0.5 }}>
                    <Tooltip title={drawing.visible ? 'Hide' : 'Show'}>
                      <IconButton
                        size="small"
                        onClick={() => toggleVisibility(drawing.id)}
                      >
                        {drawing.visible ? <Visibility /> : <VisibilityOff />}
                      </IconButton>
                    </Tooltip>
                    
                    <Tooltip title="Duplicate">
                      <IconButton
                        size="small"
                        onClick={() => duplicateDrawing(drawing.id)}
                      >
                        <ContentCopy />
                      </IconButton>
                    </Tooltip>
                    
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        onClick={() => onDrawingRemove(drawing.id)}
                        color="error"
                      >
                        <Delete />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      {/* Style Popover */}
      <Popover
        open={Boolean(stylePopoverAnchor)}
        anchorEl={stylePopoverAnchor}
        onClose={() => setStylePopoverAnchor(null)}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'left',
        }}
      >
        <Box sx={{ p: 3, minWidth: 250 }}>
          <Typography variant="subtitle2" gutterBottom>
            Drawing Style
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                label="Color"
                type="color"
                value={currentStyle.color}
                onChange={(e) => setCurrentStyle(prev => ({
                  ...prev,
                  color: e.target.value
                }))}
                size="small"
                fullWidth
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="body2" gutterBottom>
                Line Width: {currentStyle.lineWidth}px
              </Typography>
              <Slider
                value={currentStyle.lineWidth}
                onChange={(_, value) => setCurrentStyle(prev => ({
                  ...prev,
                  lineWidth: value as number
                }))}
                min={1}
                max={10}
                step={1}
                marks
                size="small"
              />
            </Grid>
            
            <Grid item xs={12}>
              <FormControl size="small" fullWidth>
                <InputLabel>Line Style</InputLabel>
                <Select
                  value={currentStyle.lineStyle}
                  label="Line Style"
                  onChange={(e) => setCurrentStyle(prev => ({
                    ...prev,
                    lineStyle: e.target.value as any
                  }))}
                >
                  <MenuItem value="solid">Solid</MenuItem>
                  <MenuItem value="dashed">Dashed</MenuItem>
                  <MenuItem value="dotted">Dotted</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            {(selectedTool === 'rectangle' || selectedTool === 'circle') && (
              <>
                <Grid item xs={12}>
                  <TextField
                    label="Fill Color"
                    type="color"
                    value={currentStyle.fillColor?.replace(/[0-9A-Fa-f]{2}$/, '') || '#2196F3'}
                    onChange={(e) => setCurrentStyle(prev => ({
                      ...prev,
                      fillColor: e.target.value + Math.round((prev.fillOpacity || 0.2) * 255).toString(16).padStart(2, '0')
                    }))}
                    size="small"
                    fullWidth
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Typography variant="body2" gutterBottom>
                    Fill Opacity: {Math.round((currentStyle.fillOpacity || 0.2) * 100)}%
                  </Typography>
                  <Slider
                    value={currentStyle.fillOpacity || 0.2}
                    onChange={(_, value) => setCurrentStyle(prev => ({
                      ...prev,
                      fillOpacity: value as number
                    }))}
                    min={0}
                    max={1}
                    step={0.1}
                    size="small"
                  />
                </Grid>
              </>
            )}
          </Grid>
          
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              size="small"
              onClick={() => setStylePopoverAnchor(null)}
            >
              Close
            </Button>
          </Box>
        </Box>
      </Popover>
    </Box>
  );
};

export default DrawingTools;
