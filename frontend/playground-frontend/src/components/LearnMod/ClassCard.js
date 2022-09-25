import React from "react";
import PropTypes from 'props-type';
import { useNavigate } from "react-router-dom";

const ClassCard = (props) => 
{
    const points = props.info.points;
    let completed = false;
    if (props.info.points === 100) {
        completed = true;
    }

    const navigate = useNavigate();

    return(
            <div className="class" style={completed ? {border:"3px solid blue"} : {}} >
                <div className="classHeader">
                    <h3 id="classTitle"> {props.info.title} </h3>
                    <div className="circleWrap">
                        <div className="circleFill" style={completed ? {background:` conic-gradient(blue ${360*points/100}deg, #ededed 0deg)`}: {background:` conic-gradient(var(--primary) ${360*points/100}deg, #ededed 0deg)`}}>
                            <div className="circleInFill"> 
                            <span className="textInCircle">{points}%</span>
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
                                            <div className="progressContainer" id="subClassBar" style={completed ? {border:"3px solid blue"} : {}}> 
                                                <div className="progressFill" style={completed ? {backgroundColor:"blue", width:`${subClass.points}%`} : {width:`${subClass.points}%`} }></div>
                                            </div>
                                        </div>
                                           );
                                }
                        )}
                    </div>
                </div>
                <div className='classFooter'>
                    <button id="classBtn" onClick={() => navigate("/LearnContent", {state: props.info})} style={completed ? {border:"3px solid blue"} : {}} >{completed ? "Completed": "Start"}</button>
                </div>
            </div>
    );
};

const propTypes = {info:PropTypes.object};
ClassCard.propTypes = propTypes;

export default ClassCard;
