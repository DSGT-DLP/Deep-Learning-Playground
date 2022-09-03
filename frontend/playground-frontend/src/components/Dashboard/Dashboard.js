import React from "react";
import PropTypes from 'props-type';

const Dashboard = (props) => 
{

    let completed = false;
    if (props.info.points === 100) {
        completed = true;
    }

    return(
        <div className="dashboardRow">
            <div className="classTitle">
            <p>{props.info.title}</p>
            </div>
            <div className="progressContainer" id="dashboardBar" style={completed ? {width:"25%", border:"3px solid blue" } :{width:"25%"}}> 
                <div className="progressFill" style={completed ? {width:` ${props.info.points}%`, background:"blue" } :{width:` ${props.info.points}%` }}></div>
            </div>
        </div>
    );
};

const propTypes = {info:PropTypes.object};
Dashboard.propTypes = propTypes;

export default Dashboard;
