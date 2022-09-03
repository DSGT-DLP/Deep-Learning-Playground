import React from 'react';
import ClassCard from './ClassCard';

const LearnMod = () => {

    const class1 = {title:"Intro to Machine Learning",
                    points:100,
                    subClasses:[{title:"Regression",points:100},
                               {title:"Classification",points:100},
                               {title:"Modeling",points:100}]
                    };
    const class2 = {title:"Intro to Machine Learning",
                    points:33,
                    subClasses:[{title:"Regression",points:100},
                               {title:"Classification",points:0},
                               {title:"Modeling",points:0}]
                    };
    const class3 = {title:"Intro to Machine Learning",
                    points:0,
                    subClasses:[{title:"Regression",points:0},
                               {title:"Classification",points:0},
                               {title:"Modeling",points:0}]
                    };

     const lessons = [class1,class1,class1,class2,class3];               
    
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
