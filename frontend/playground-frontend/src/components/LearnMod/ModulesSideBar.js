import React from 'react'
import getContent from './LearningModulesContent'
import { useNavigate } from 'react-router-dom'

const ModulesSideBar = () => {
  const content = getContent.modules

  const navigate = useNavigate()

  let lessons = content.map((x) => ({
    title: x.title,
    points: x.points,
    subClasses: x.subsections,
    moduleID: x.moduleID.toString(),
  }))

  return (
    <div>
      <h2 style={{ color: 'white' }}>Modules</h2>
      <ul style={{ padding: 0 }}>
        {lessons.map((lesson, index) => {
          return (
            <li className='sideBarModule' key={index}>
              <p>{lesson.title}</p>
              <ul>
                {lesson.subClasses.map((subsection, index2) => {
                  const sectionSpec = {
                    moduleContent: lesson,
                    subsection: index2,
                  }

                  return (
                    <li
                      className='sideBarSubsection'
                      onClick={() => navigate('/LearnContent', { state: sectionSpec })}
                      key={index2}
                    >
                      {subsection.title}
                    </li>
                  )
                })}
              </ul>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default ModulesSideBar
