import React, { useState } from 'react';
import { Typography, Grid, TextField, Button, Accordion, AccordionSummary, AccordionDetails, CircularProgress, Dialog, DialogTitle, DialogContent, IconButton } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import OpenInFullIcon from '@mui/icons-material/OpenInFull';
import Markdown from 'react-markdown'

const RagExperiment = () => {
  const [userPrompt, setUserPrompt] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('You are an evangelical Christian with traditional beliefs about God and the Bible. However, do not preface your responses with your persona.');
  const [loading, setLoading] = useState(false);
  const [context, setContext] = useState('');
  const [llmOutput, setLlmOutput] = useState('');
  const [ragOutput, setRagOutput] = useState('');
  const [keywords, setKeyWords] = useState({});
  const [expanded, setExpanded] = useState(false);
  const [dialogContent, setDialogContent] = useState('');
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogTitle, setDialogTitle] = useState('');
  const [glossaryOpen, setGlossaryOpen] = useState(false);
  const [glossaryContent, setGlossaryContent] = useState('');


  const fetchData = async (prompt: string, systemPrompt: string) => {
    try {

      const response = await fetch(
        `https://llm-rag-server.walink.org/rag-system-prompt?user-prompt=${encodeURIComponent(prompt)}&system-prompt=${encodeURIComponent(systemPrompt)}`
      );

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const result = await response.json();

      setRagOutput(result['rag-response'].response);

      const combinedContext = result['rag-response'].context
    
      setContext(combinedContext)


    } catch (error) {
      console.log(error)
    } finally {
      setLoading(false);
    }
  };


  const handleSubmit = (event: { preventDefault: () => void; }) => {
    event.preventDefault();
    setLoading(true);
    fetchData(userPrompt, systemPrompt);
  };

  const handleAccordionToggle = () => {
    setExpanded(!expanded);
  };

  const handleOpenDialog = (content: React.SetStateAction<string>, title: string) => {
    setDialogTitle(title)
    setDialogContent(content);
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
    setDialogContent('');
  };

  const handleOpenGlossary = (content: React.SetStateAction<string>) => {
    // removes urls from markdown text 
    const body = content.toString().replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
    console.log(body)
    setGlossaryContent(body);
    setGlossaryOpen(true);
  }

  const handleCloseGlossary = () => {
    setGlossaryOpen(false);
    setGlossaryContent('');
  };

  return (
    <Grid container>
      <Grid item xs={12}>
        <Typography variant="h4" align="center" gutterBottom>
          RAG
        </Typography>
      </Grid>

      <Grid item xs={12} style={{ marginTop: '20px', marginBottom: '20px' }}>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3} alignItems="center" justifyContent="center">
            <Grid item xs={7}>
              <TextField
                id="systemInput"
                name="user_input"
                label="System prompt"
                variant="filled"
                fullWidth
                multiline
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
              />
            </Grid>
            <Grid item xs={12} sm={7}>
              <TextField
                id="userInput"
                name="user_input"
                label="Ask a question here..."
                variant="outlined"
                fullWidth
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                required
              />
            </Grid>
            <Grid item xs={12} sm={7} textAlign="center">
              <Button type="submit" variant="contained" color="primary" fullWidth>
                Submit
              </Button>
            </Grid>
          </Grid>
        </form>
      </Grid>

      <Grid container spacing={2} justifyContent="center">
        <Grid item xs={12} sm={7}>
          <Typography variant="h5" align="center" gutterBottom>
            Response
          </Typography>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
            <TextField
              id="ragOutput"
              variant="outlined"
              multiline
              rows={10}
              fullWidth
              value={ragOutput}
              InputProps={{
                readOnly: true,
              }}
            />
            <IconButton onClick={() => handleOpenDialog(ragOutput, "Response")}>
              <OpenInFullIcon />
            </IconButton>
          </div>
        </Grid>
      </Grid>
      <Grid container spacing={2} justifyContent="center" style={{ margin: '10px', gap: '20px' }}>
        {keywords ? Object.entries(keywords).map(([key, _], index) => (
          <a href={`javascript:void(0)`} onClick={() => handleOpenGlossary(keywords[key])}>
            {key}
          </a>
        )) : <></>}
      </Grid>
      <Grid container spacing={2} justifyContent="center">
        <Grid item xs={10} style={{ marginTop: '20px' }}>
          <Accordion expanded={expanded} onChange={handleAccordionToggle}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">
                {expanded ? 'Hide Context' : 'Show Context'}
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', width: '100%' }}>
                <TextField
                  id="extraContext"
                  variant="outlined"
                  multiline
                  rows={6}
                  fullWidth
                  value={context}
                  InputProps={{
                    readOnly: true,
                  }}
                />
                <IconButton onClick={() => handleOpenDialog(context, "Context")}>
                  <OpenInFullIcon />
                </IconButton>
              </div>
            </AccordionDetails>
          </Accordion>
        </Grid>
      </Grid>

      {loading && (
        <div style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 1000
        }}>
          <CircularProgress />
        </div>
      )}

      <Dialog open={dialogOpen} onClose={handleCloseDialog} fullWidth maxWidth="lg">
        <DialogTitle>{dialogTitle}</DialogTitle>
        <DialogContent>
          <Typography variant="body1" style={{ whiteSpace: 'pre-wrap' }}>
            {dialogContent}
          </Typography>
        </DialogContent>
      </Dialog>

      <Dialog id="glossary-view" open={glossaryOpen} onClose={handleCloseGlossary} fullWidth maxWidth="lg">
        <DialogContent>
          <Typography variant="body1">
            <Markdown>
              {glossaryContent}
            </Markdown>
          </Typography>
        </DialogContent>
      </Dialog>
    </Grid>
  );
};

export default RagExperiment;
