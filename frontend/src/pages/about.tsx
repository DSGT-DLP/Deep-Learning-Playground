import React, { type ReactNode } from "react";
import { URLs } from "../constants";
import pythonLogo from "../../public/images/logos/python-logo.png";
import pandasLogo from "../../public/images/logos/pandas-logo.png";
import pyTorchLogo from "../../public/images/logos/pytorch-logo.png";
import flaskLogo from "../../public/images/logos/flask-logo.png";
import reactLogo from "../../public/images/logos/react-logo.png";
import awsLogo from "../../public/images/logos/aws-logo.png";
import Image from "next/image";
import NavbarMain from "@/common/components/NavBarMain";
import Footer from "@/common/components/Footer";

const urlOpener = (url: string) => () => window.open(url);

function renderHeading(): ReactNode {
  /**
   * Function that renders the heading of the About Page
   */
  return (
    <div id="header-section" data-testid="header">
      <h1 className="headers">Deep Learning Playground</h1>
      <h2>Your destination for training Deep Learning models</h2>
    </div>
  );
}

function renderMotivationSection(): ReactNode {
  /**
   * Section in About page that explains motivation behind DLP
   */
  return (
    <div className="sections" id="motivation" data-testid="motivation">
      <h2>Motivation</h2>
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
          Active development with support and functionality being enhanced by a
          dedicated, user-focused team of software engineers
        </li>
      </ul>
    </div>
  );
}

function renderTechnologiesUsedSection(): ReactNode {
  /**
   * Explain what technologies we use for DLP
   */
  return (
    <div className="sections" id="tech" data-testid="tech">
      <h2>Technologies Used</h2>

      <div className="tech-rows">
        <div className="tech-row-content" data-testid="tech-row-content">
          <Image
            src={pythonLogo}
            className="tech-img-content"
            onClick={urlOpener("https://docs.python.org/3/")}
            alt="Python logo"
          />
          <p>
            <b>Python:</b> Programming language widely used for Data Science,
            AI, and ML. Easily understandable by a human and has a lot of
            developer support
          </p>
        </div>
        <div className="tech-row-content" data-testid="tech-row-content">
          <Image
            src={pyTorchLogo}
            className="tech-img-content"
            onClick={urlOpener("https://pytorch.org/docs/stable/index.html")}
            alt="Pytorch logo"
          />
          <p>
            <b>Pytorch:</b> Libary that helps build, train, test deep learning
            models. We use this library to build the user-defined deep learning
            model
          </p>
        </div>
        <div className="tech-row-content" data-testid="tech-row-content">
          <Image
            src={pandasLogo}
            className="tech-img-content"
            onClick={urlOpener("https://pandas.pydata.org/docs/")}
            alt="pandas logo"
          />
          <p>
            <b>pandas:</b> Python library that allows one to parse CSV files and
            extract relevant information. It's very user friendly and has
            helpful documentation
          </p>
        </div>
        <div className="tech-row-content" data-testid="tech-row-content">
          <Image
            src={flaskLogo}
            className="tech-img-content"
            onClick={urlOpener("https://flask.palletsprojects.com/en/2.1.x/")}
            alt="Flask logo"
          />
          <p>
            <b>Flask:</b> Backend service that allows for the modeling magic to
            happen
          </p>
        </div>
        <div className="tech-row-content" data-testid="tech-row-content">
          <Image
            src={reactLogo}
            className="tech-img-content"
            onClick={urlOpener("https://reactjs.org/docs/getting-started.html")}
            alt="React logo"
          />
          <p>
            <b>React:</b> JavaScript library used to display the website to the
            user
          </p>
        </div>
        <div className="tech-row-content" data-testid="tech-row-content">
          <Image
            src={awsLogo}
            className="tech-img-content"
            onClick={urlOpener("https://docs.aws.amazon.com/")}
            alt="AWS logo"
          />
          <p>
            <b>AWS:</b> Amazon Web Services provides on-demand cloud computing
            platforms and APIs to this project on a metered pay-as-you-go basis
          </p>
        </div>
      </div>
    </div>
  );
}

function renderInstallInstructions(): ReactNode {
  /**
   * Explain how to setup Deep Learning Playground on your local machine for
   * those interested in developing
   */

  return (
    <div className="sections" id="installation" data-testid="installation">
      <h2>Installation</h2>
      <p>
        See the README.md in the <a href={URLs.github}>Github Repo</a> for setup
        instructions. These setup instructions are mainly for those
        developing/enhancing the tool. For the user, simply go to the "Deep
        Learning tab" at the top of the page.
      </p>
    </div>
  );
}

function renderUserInstructions(): ReactNode {
  /**
   * Explain what the user will have to do to use DLP
   */
  return (
    <div className="sections" id="user" data-testid="user">
      <h2>User</h2>
      <p>
        As the user, all you need to do in the Deep Learning Playground is
        upload your dataset or enter a URL to it (dataset must be in a CSV or
        ZIP file format). Drag and drop the layers from the available blocks to
        the purple "+" icon, select values in the dropdowns, and click train!
        Sit back and relax and let us take care of building the model!
      </p>

      <p>
        Once the model building has finished, you will get a downloadable image
        corresponding to the plots of loss (and accuracy depending on if you do
        a classification or regression problem). You will also have access to a
        CSV corresponding to the performance stats/metrics of your model!
      </p>
    </div>
  );
}

function renderDeveloperInstructions(): ReactNode {
  /**
   * Render instructions for developers on DLP
   */
  return (
    <div className="sections" id="dev" data-testid="dev">
      <h2>Development</h2>
      <p data-testid="developer-greeting">
        Want to contribute? Great! Email one of the collaborators in the Github
        Repo for more information on how you can get involved in pushing the
        impact of Deep Learning Playground Forward.
      </p>
      <p>
        If you have any problems with using the tool or have a request for a
        future feature to the tool, please post an issue in GitHub by clicking
        on the "Issues" tab at the top and selecting "New Issue". The
        development team will respond to your request ASAP!
      </p>
    </div>
  );
}

function renderLicense(): ReactNode {
  /**
   * Render License for DLP
   */
  return (
    <div className="sections" id="license">
      <h2>License</h2>
      <p>MIT</p>
    </div>
  );
}

const About = () => {
  return (
    <>
      <NavbarMain />
      <div id="about">
        {renderHeading()}

        {renderMotivationSection()}

        {renderTechnologiesUsedSection()}

        {renderInstallInstructions()}

        {renderUserInstructions()}

        {renderDeveloperInstructions()}

        {renderLicense()}
      </div>
      <Footer />
    </>
  );
};

export default About;
