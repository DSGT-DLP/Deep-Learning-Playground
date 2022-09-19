import React from 'react';
import ClassCard from './ClassCard';
import getContent from './LearningModulesContent';

const LearnMod = () => {

    const content = getContent.modules;
    
    let lessons = content.map(x => (
        {
            title: x.title,
            points: x.points,
            subClasses: x.subsections
        }
    ));

    console.log(lessons[0]);
    
    return (
        <>
      <div id="header-section">
            <h1 className="headers">Your learning module Mateo!</h1>
      </div>

     <div id="learningBody">
        <div className='classes'>
            {
                lessons.map( (lesson, index) => 
                {
                
                    return(
                        <ClassCard info={lesson} key={index}/>
                    );
                }
                )
            }
        </div>
        
      </div>   
        </>
    );
};

export default LearnMod;
