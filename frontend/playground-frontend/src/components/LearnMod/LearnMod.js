import React from 'react';

const LearnMod = () => {
    return (
        <>
        <div id="header-section">
            <h1 className="headers">Your learning module Mateo!</h1>
      </div>

     <div id="learningBody">
        <div className="progress">
            <h1>Progress</h1>
            <p>Class 1 40/100 points</p>
            <p>Class 1 40/100 points</p>
            <p>Class 1 40/100 points</p>
            <p>Class 1 40/100 points</p>
            <p>Class 1 40/100 points</p>
        </div>
        <div className='classes'>
            <div className="class" >
                <div className="classHeader">
                    <h3 id="classTitle">Intro to Machine Learning</h3>
                    <div className='progressBar' style={{width:"40%"}}>
                        <h3 id="headerProgress">50/100 points</h3>
                        <div className="progressContainer" style={{width:"50%"}}> 
                            <div className="progressFill"></div>
                        </div>
                    </div>
                </div>
                <div className="classBody">
                    <div className='classBodyLeft'>
                        <p id="classBodyText">Part 1 Regression model</p>
                        <p id="classBodyText">Part 2 Regression model</p>
                        <p id="classBodyText">Part 3 Regression model</p>
                    </div>
                    <div className='classBodyRight'>
                        <p id="classBodyText">0/50 points</p>
                        <p id="classBodyText">0/50 points</p>
                        <p id="classBodyText">0/50 points</p>
                    </div>
                </div>
                <div className='classFooter'>
                    <button id="classBtn">Start</button>
                </div>
            </div>
            <div className="class" >
                <div className="classHeader">
                    <h3 id="classTitle">Intro to Machine Learning</h3>
                    <div className='progressBar' style={{width:"40%"}}>
                        <h3 id="headerProgress">50/100 points</h3>
                        <div className="progressContainer" style={{width:"50%"}}> 
                            <div className="progressFill"></div>
                        </div>
                    </div>
                </div>
                <div className="classBody">
                    <div className='classBodyLeft'>
                        <p id="classBodyText">Part 1 Regression model</p>
                        <p id="classBodyText">Part 2 Regression model</p>
                        <p id="classBodyText">Part 3 Regression model</p>
                    </div>
                    <div className='classBodyRight'>
                        <p id="classBodyText">0/50 points</p>
                        <p id="classBodyText">0/50 points</p>
                        <p id="classBodyText">0/50 points</p>
                    </div>
                </div>
                <div className='classFooter'>
                    <button id="classBtn">Start</button>
                </div>
            </div>
        </div>
        
      </div>   

        </>
    );
};

export default LearnMod;
