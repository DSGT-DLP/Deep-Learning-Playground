import NavbarMain from "@/common/components/NavBarMain";
import React from "react";
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

const TrainSpace = () => {
  const blocks = [
    { id: 1, content: 'Block 1' },
    { id: 2, content: 'Block 2' },
    { id: 3, content: 'Block 3' },
    // Add more blocks as needed
  ];

  return (
    <DndProvider backend={HTML5Backend}>
      <div>
        <NavbarMain />
        {/* Menubar */}
        <div style={{ 
          width: '100%', 
          height: '50px', 
          backgroundColor: '#808080',
        }}>
        </div>
        <div style={{ display: 'flex' }}>
          {/* Sidebar */}
          <div style={{
            width: '200px',
            height: 'calc(100vh - 60px)', // Adjusted height to consider the menu bar
            position: 'fixed',
            backgroundColor: '#f5f5f5',
            top: '64px' // Adjusted top value
          }}>
            <p>Sidebar content will go here</p>
          </div>
          <div style={{ marginLeft: '200px' }}>
            <h1>Welcome to TrainSpace</h1>
            {/* Draggable List */}
            <div>
              {blocks.map((block) => (
                <div key={block.id} style={{ border: '1px solid #ccc', margin: '5px', padding: '10px', cursor: 'move' }}>
                  {block.content}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </DndProvider>
  );
};

export default TrainSpace;
