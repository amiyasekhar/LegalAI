import React, { useEffect, useState } from 'react';

/* global Word */

export default function App() {
  const [paragraphs, setParagraphs] = useState([]);
  const [selectedText, setSelectedText] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    Office.onReady((info) => {
      if (info.host === Office.HostType.Word) {
        // Set up document selection change event
        Office.context.document.addHandlerAsync(
          Office.EventType.DocumentSelectionChanged,
          getSelectedText
        );
      }
    });

    // Cleanup event handler on component unmount
    return () => {
      Office.context.document.removeHandlerAsync(
        Office.EventType.DocumentSelectionChanged,
        getSelectedText
      );
    };
  }, []);

  const getSelectedText = async () => {
    try {
      await Word.run(async (context) => {
        const selection = context.document.getSelection();
        selection.load('text');
        await context.sync();
        setSelectedText(selection.text);
      });
    } catch (error) {
      console.error('Error getting selected text:', error);
    }
  };

  const highlightParagraphs = async () => {
    setErrorMessage('');
    try {
      await Word.run(async (context) => {
        console.log('Starting paragraph highlighting...');
        
        const documentBody = context.document.body;
        const paragraphItems = documentBody.paragraphs;
        paragraphItems.load(['text', 'style']);
        
        await context.sync();
        console.log('Found paragraphs:', paragraphItems.items.length);
        
        const paragraphTexts = paragraphItems.items.map(p => p.text);
        setParagraphs(paragraphTexts);
        
        paragraphItems.items.forEach((paragraph, index) => {
          try {
            paragraph.getBorder(Word.BorderLocation.Top).setColor('red');
            paragraph.getBorder(Word.BorderLocation.Bottom).setColor('red');
            paragraph.getBorder(Word.BorderLocation.Left).setColor('red');
            paragraph.getBorder(Word.BorderLocation.Right).setColor('red');
            
            paragraph.getBorder(Word.BorderLocation.Top).setWidth(1.5);
            paragraph.getBorder(Word.BorderLocation.Bottom).setWidth(1.5);
            paragraph.getBorder(Word.BorderLocation.Left).setWidth(1.5);
            paragraph.getBorder(Word.BorderLocation.Right).setWidth(1.5);
            
            console.log(`Highlighted paragraph ${index + 1}`);
          } catch (paraError) {
            console.error(`Error highlighting paragraph ${index + 1}:`, paraError);
          }
        });
        
        await context.sync();
        console.log('Highlighting complete');
      });
    } catch (error) {
      console.error('Error in highlightParagraphs:', error);
      setErrorMessage('Error highlighting paragraphs. Please check console for details.');
    }
  };

  return (
    <div className="ms-welcome">
      <header className="ms-welcome__header ms-bgColor-neutralLighter">
        <h1 className="ms-font-su">Paragraph Highlighter</h1>
      </header>
      <main className="ms-welcome__main">
        <button 
          className="ms-Button ms-Button--primary"
          onClick={highlightParagraphs}
        >
          <span className="ms-Button-label">Highlight All Paragraphs</span>
        </button>

        {errorMessage && (
          <div style={{ color: 'red', margin: '10px 0' }}>
            {errorMessage}
          </div>
        )}

        {selectedText && (
          <div 
            className="ms-Grid-row"
            style={{ 
              marginTop: '20px', 
              padding: '15px', 
              backgroundColor: '#f0f0f0', 
              border: '1px solid #dedede',
              borderRadius: '4px'
            }}
          >
            <div className="ms-fontWeight-semibold">Selected Text:</div>
            <div style={{ fontSize: '12px', marginTop: '5px' }}>{selectedText}</div>
          </div>
        )}

        <div className="ms-welcome__main" style={{ marginTop: '20px' }}>
          <div className="ms-fontWeight-semibold">All Paragraphs:</div>
          {paragraphs.map((text, index) => (
            <div 
              key={index} 
              className="ms-Grid-row"
              style={{ marginTop: '10px', padding: '10px', border: '1px solid #edebe9' }}
            >
              <div className="ms-fontWeight-semibold">Paragraph {index + 1}</div>
              <div style={{ fontSize: '12px' }}>{text}</div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}