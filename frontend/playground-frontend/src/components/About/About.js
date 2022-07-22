import React from "react";
import pythonLogo from "../../images/logos/python-logo.png";
import pandasLogo from "../../images/logos/pandas-logo.svg";
import pyTorchLogo from "../../images/logos/pytorch-logo.png";
import flaskLogo from "../../images/logos/flask-logo.png";
import reactLogo from "../../images/logos/react-logo.png";
import awsLogo from "../../images/logos/aws-logo.png";

const About = () => {
  const urlOpener = (url) => () => window.open(url);

  return (
    <div className="about-page">
      <div id="header-section">
        <h1 className="headers">Deep Learning Playground</h1>
        <h2>Your destination for training Deep Learning models</h2>
      </div>

      <div className="sections" id="motivation">
        <h3>Motivation</h3>
        <p>
          Deep Learning has made advancements recently. People are trying to
          utilize Deep Learning to build models to solve increasingly complex
          problems and drive business impact. Two well known libraries for deep
          learning are Pytorch and Tensorflow. While these open source libraries
          make deep learning accessible, to use them, one requires prior coding
          experience. However, we believe that the low code, no code movement is
          picking up and if we want to improve accessibility and empower the
          average person to delve into deep learning, we should build an online
          "playground:" Deep Learning Playground (DLP).
        </p>
        <p>
          Deep Learning Playground (DLP) allows for the average user to upload a
          dataset, set hyper-parameters, and drag/drop layers and build a deep
          learning model without any prior programming knowledge.
        </p>
        <h3>Key Takeaways:</h3>
        <ul>
          <li>Easy environment to prototype your deep learning models</li>
          <li>No need to write code for deep learning</li>
          <li>Interactive tool to learn about deep learning</li>
          <li>
            Active development with support and functionality being enhanced by
            a dedicated, user-focused team of software engineers
          </li>
        </ul>
      </div>

      <div className="sections" id="tech">
        <h3>Technologies Used</h3>

        <div className="tech-rows">
          <div className="tech-row-content">
            <img
              src={pythonLogo}
              className="tech-img-content"
              onClick={urlOpener("https://docs.python.org/3/")}
              alt="Python logo"
            />
            <span className="tech-text-content">
              <b>Python:</b> Programming language widely used for Data Science,
              AI, and ML. Easily understandable by a human and has a lot of
              developer support
            </span>
          </div>
          <div className="tech-row-content">
            <img
              src={pyTorchLogo}
              className="tech-img-content"
              onClick={urlOpener("https://pytorch.org/docs/stable/index.html")}
              alt="Pytorch logo"
            />
            <b>Pytorch:</b> Libary that helps build, train, test deep learning
            models. We use this library to build the user-defined deep learning
            model
          </div>
          <div className="tech-row-content">
            <img
              src={pandasLogo}
              className="tech-img-content"
              onClick={urlOpener("https://pandas.pydata.org/docs/")}
              alt="pandas logo"
            />
            <b>pandas:</b> Python library that allows one to parse CSV files and
            extract relevant information. It's very user friendly and has
            helpful documentation
          </div>
          <div className="tech-row-content">
            <img
              src={flaskLogo}
              className="tech-img-content"
              onClick={urlOpener("https://flask.palletsprojects.com/en/2.1.x/")}
              alt="Flask logo"
            />
            <b>Flask:</b> Backend service that allows for the modeling magic to
            happen
          </div>
          <div className="tech-row-content">
            <img
              src={reactLogo}
              className="tech-img-content"
              onClick={urlOpener(
                "https://reactjs.org/docs/getting-started.html"
              )}
              alt="React logo"
            />
            <b>React:</b> JavaScript library used to display the website to the
            user
          </div>
          <div className="tech-row-content">
            <img
              src={awsLogo}
              className="tech-img-content"
              onClick={urlOpener("https://docs.aws.amazon.com/")}
              alt="AWS logo"
            />
            <b>AWS:</b> Amazon Web Services provides on-demand cloud computing
            platforms and APIs to this project on a metered pay-as-you-go basis
          </div>
        </div>
      </div>

      <div className="sections" id="installation">
        <h3>Installation</h3>
        <p>
          See the README.md in the{" "}
          <a href="https://github.com/karkir0003/Deep-Learning-Playground">
            Github Repo
          </a>{" "}
          for setup instructions. These setup instructions are mainly for those
          developing/enhancing the tool. For the user, simply go to the "Deep
          Learning tab" at the top of the page.
        </p>
      </div>

      <div className="sections" id="user">
        <h3>User</h3>
        <p>
          As the user, all you need to do in the Deep Learning Playground is
          upload your dataset or enter a URL to it (dataset must be in a CSV or
          ZIP file format). Drag and drop the layers from the available blocks
          to the purple "+" icon, select values in the dropdowns, and click
          train! Sit back and relax and let us take care of building the model!
        </p>

        <p>
          Once the model building has finished, you will get a downloadable
          image corresponding to the plots of loss (and accuracy depending on if
          you do a classification or regression problem). You will also have
          access to a CSV corresponding to the performance stats/metrics of your
          model!
        </p>
      </div>

      <div className="sections" id="dev">
        <h3>Development</h3>
        <p>
          Want to contribute? Great! Email one of the collaborators in the
          Github Repo for more information on how you can get involved in
          pushing the impact of Deep Learning Playground Forward.
        </p>
        <p>
          If you have any problems with using the tool or have a request for a
          future feature to the tool, please post an issue in GitHub by clicking
          on the "Issues" tab at the top and selecting "New Issue". The
          development team will respond to your request ASAP!
        </p>
      </div>

      <div className="sections" id="license">
        <h3>License</h3>
        MIT
      </div>
      {/* <footer/> */}
    </div>
  );
};

export default About;
