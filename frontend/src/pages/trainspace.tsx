import NavbarMain from "@/common/components/NavBarMain";
import React from "react";

const TrainSpace = () => {
    return (
        <div>
            <NavbarMain />
            //Menubar
            <div style={{ 
                width: '100%', 
                height: '50px', 
                backgroundColor: '#808080',
            }}>
            </div>
            <div style={{ display: 'flex' }}>
                //Sidebar
                <div style={{
                    width: '200px',
                    height: 'calc(100vh - 60px)',
                    position: 'fixed',
                    backgroundColor: '#f5f5f5',
                    top: '64px'
                }}>
                    <p>Sidebar content will go here</p>
                </div>
                <div style={{ marginLeft: '200px' }}>
                    <h1>Welcome to TrainSpace</h1>
                </div>
            </div>
        </div>
    );
};

export default TrainSpace;