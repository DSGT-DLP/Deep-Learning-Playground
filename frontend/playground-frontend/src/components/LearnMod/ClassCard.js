import React from "react";
import PropTypes from 'prop-types';
import { useNavigate } from "react-router-dom";

const ClassCard = (props) => 
{
    const points = props.info.points;
    let pointsEarned = 0;
    if (props.moduleProgress != null) {
        pointsEarned = props.moduleProgress[props.moduleID].modulePoints;
    }

    let completed = false;
    if (pointsEarned === points) {
        completed = true;
    }

    const navigate = useNavigate();

    return(
            <div className="class" style={completed ? {border:"3px solid green"} : {}} >
                <div className="classHeader">
                    <h3 id="classTitle"> {props.info.title} </h3>
                    <div className="circleWrap">
                        <div className="circleFill" style={completed ? {background:` conic-gradient(green ${360*pointsEarned/(points)}deg, #ededed 0deg)`}: {background:` conic-gradient(var(--primary) ${360*pointsEarned/(points)}deg, #ededed 0deg)`}}>
                            <div className="circleInFill"> 
                            <span className="textInCircle">{pointsEarned/points}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="classBody">
                    <div className='classBodyLeft'>
                        {
                            props.info.subClasses.map( (subClass,index) => 
                                {
                                    return(
                                            <p id="classBodyText" key={index}>{subClass.title}</p>
                                           );
                                }
                        )}
                    </div>
                    <div className='classBodyRight'>
                        {
                            props.info.subClasses.map( (subClass,index) => 
                                {
                                    return(
                                        <div className='progressBar' key={index} id="subClassBar" >
                                            <div className="progressContainer" id="subClassBar" style={completed ? {border:"3px solid green"} : {}}> 
                                                <div className="progressFill" style={completed ? {backgroundColor:"green", width:`${subClass.points}%`} : {width:`${subClass.points}%`} }></div>
                                            </div>
                                        </div>
                                           );
                                }
                        )}
                    </div>
                </div>
                <div className='classFooter'>
                    <button id="classBtn" onClick={() => navigate("/LearnContent", {state: props.info})} style={completed ? {border:"3px solid green"} : {}} >{completed ? "Completed": "Start"}</button>
                </div>
            </div>
    );
};

const propTypes = {info:PropTypes.object, moduleProgress:PropTypes.object, moduleID: PropTypes.string};
ClassCard.propTypes = propTypes;

export default ClassCard;
