import React from 'react';
import Dashboard from './Dashboard';

const DashboardPage = () => {

  const class1 = {title:"Intro to Machine Learning",
  points:100,
  };

const class2 = {title:"Intro to Deep Learning",
  points:33,
  };

  const class3 = {title:"Convolutional layers",
  points:0,
  };

const lessons = [class1,class1,class1,class2,class3]; 
let averagePoints = 0;

lessons.forEach(lesson => {
  averagePoints += lesson.points;
});
averagePoints = averagePoints/lessons.length;

    return (
    <>
      <div id="header-section">
            <h1 className="headers">Your Dashboard</h1>
      </div>

     <div id="learningBody">
          <div className="dashboard">
            <div className="dashboardHeader">
              <h1>Dashboard</h1>
              <div className="circleWrap">
                        <div className="circleFill" style={{background:` conic-gradient(var(--primary) ${360*averagePoints/100}deg, #ededed 0deg)`}}>
                            <div className="circleInFill"> 
                            <span className="textInCircle">{averagePoints}%</span>
                            </div>
                        </div>
             </div>
             </div >
              {
                  lessons.map( (lesson,index) => 
                  {  
                    return (   
                            <Dashboard info={lesson} key={index}/>
                            );
                  }) 
              };
           </div>
      </div>   
    </>
    );
};

export default DashboardPage;
