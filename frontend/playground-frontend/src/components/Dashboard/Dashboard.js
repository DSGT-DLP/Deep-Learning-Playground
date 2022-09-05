import React from 'react';
import PropTypes from 'prop-types';
import DSGTLogo from "../../images/logos/dlp_branding/dlp-logo.png";

const Dashboard = props => {
    return (
        <div style="height: 100vh; ">
            <img src={DSGTLogo} alt="DSGT Logo" width="60" height="60" />
            <h1>Deep Learning Playground</h1>
            <button></button>
            hello
        </div>
    );
};

Dashboard.propTypes = {
    
};

export default Dashboard;