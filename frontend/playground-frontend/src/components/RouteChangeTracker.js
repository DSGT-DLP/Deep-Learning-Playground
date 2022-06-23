/*Google Analytics stuff. Don't worry about this file for adding stuff to the site!*/
import React from "react";
import { withRouter } from "react-router-dom";
import ReactGA from "react-ga";
const RouteChangeTracker = ({ history }) => {
  history.listen((location, action) => {
    ReactGA.set({ page: location.pathname });
    ReactGA.pageview(location.pathname);
  });

  return <div></div>;
};

export default withRouter(RouteChangeTracker);
